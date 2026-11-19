"""LAGD-Net training (paper Secs. 3 and 4) on one or several GPUs with DDP.

Phase 1, search: the weights and gate logits are trained jointly on the cross-entropy plus
lambda / N * sum_i sigmoid(g_i / T) (Eq. 2), with T annealed from 5 as 5 * 0.02 ** (e / 50).
Phase 2, threshold: at the final temperature, blocks with sigmoid(g_i / T) < tau = 0.1 keep only
their shortcut and all other gates are set to 1. The gates stay frozen from here on.
Phase 3, fine-tuning: lambda = 0; the remaining weights are fine-tuned in the same run.
"""

import logging
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lagdnet.layers import TAU, BlockGate
from lagdnet.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from lagdnet.utils import gate_state_dict, gated_blocks, get_current_temp, validate

MODEL_FACTORIES = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50,
                   'resnet101': ResNet101, 'resnet152': ResNet152}

# The recipe of Sec. 4. Fine-tuning starts with its own short linear warm-up (finetune_warmup).
DATASET_DEFAULTS = {
    'cifar10': dict(
        num_classes=10, model_dataset='cifar', input_shape=(3, 32, 32), data_root='./data',
        norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5),
        per_gpu_batch=128, base_lr=0.1, warmup_epochs=2, epochs=50,
        finetune_epochs=15, finetune_lr=0.05, finetune_warmup=2, label_smoothing=0.1),
    'tiny_imagenet': dict(
        num_classes=200, model_dataset='tiny_imagenet', input_shape=(3, 64, 64),
        data_root='./data/archive/tiny-imagenet-200',
        norm_mean=(0.4802, 0.4481, 0.3975), norm_std=(0.2302, 0.2265, 0.2262),
        per_gpu_batch=128, base_lr=0.1, warmup_epochs=5, epochs=50,
        finetune_epochs=55, finetune_lr=0.02, finetune_warmup=3, label_smoothing=0.1),
}

# Filled in by the trainers before main(); None keeps the dataset default.
CONFIG = dict(
    dataset='cifar10', model='resnet18', lambda_sparsity=None,
    epochs=None, finetune_epochs=None, per_gpu_batch=None,
    start_temp=5.0, end_temp=0.1,
    use_sync_bn=False, grad_clip_norm=2.0, seed=42, nccl_timeout_min=30,
    log_dir='./logs', ckpt_dir='./checkpoints',
)


def resolve_config(cfg):
    """Dataset defaults, overridden by the non-None entries of cfg."""
    d = dict(DATASET_DEFAULTS[cfg['dataset']])
    d.update({k: v for k, v in cfg.items() if v is not None or k not in d})
    return d


def setup_ddp(timeout_min):
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=timeout_min))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_logging(rank, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handlers = [logging.FileHandler(os.path.join(log_dir, f'rank{rank}.log'), mode='w')]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    fmt = logging.Formatter(f'%(asctime)s [rank{rank}] %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)


def set_seed(seed, rank):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_data(cfg):
    """Training loader (distributed) and the full test loader. On Tiny-ImageNet the test set is
    the labelled validation split."""
    mean, std = cfg['norm_mean'], cfg['norm_std']
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if cfg['dataset'] == 'tiny_imagenet':
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(), transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25)])
        trainset = datasets.ImageFolder(os.path.join(cfg['data_root'], 'train'), transform=train_tf)
        testset = datasets.ImageFolder(os.path.join(cfg['data_root'], 'val'), transform=test_tf)
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
        if is_main():
            datasets.CIFAR10(cfg['data_root'], train=True, download=True)
            datasets.CIFAR10(cfg['data_root'], train=False, download=True)
        dist.barrier()
        trainset = datasets.CIFAR10(cfg['data_root'], train=True, transform=train_tf)
        testset = datasets.CIFAR10(cfg['data_root'], train=False, transform=test_tf)
    sampler = DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoader(trainset, batch_size=cfg['per_gpu_batch'], sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, sampler, len(trainset), len(testset)


def build_model(cfg, local_rank):
    model = MODEL_FACTORIES[cfg['model']](dataset=cfg['model_dataset'], num_classes=cfg['num_classes'],
                                          start_temp=cfg['start_temp']).to(local_rank)
    if cfg['use_sync_bn']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return DDP(model, device_ids=[local_rank], find_unused_parameters=True)


def search_optimizer(model, lr):
    """SGD with momentum 0.9; weight decay 1e-3 on every parameter except the gate logits."""
    named = list(model.module.named_parameters())
    weights = [p for n, p in named if 'gate' not in n]
    gates = [p for n, p in named if 'gate' in n]
    return optim.SGD([{'params': weights, 'weight_decay': 1e-3},
                      {'params': gates, 'weight_decay': 0.0}], lr=lr, momentum=0.9)


def lr_at(epoch, base_lr, total, warmup):
    """Linear warm-up, then cosine decay."""
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def train_epoch(model, loader, criterion, optimizer, device, lambda_sparsity, grad_clip_norm):
    model.train()
    gates = [m for m in model.module.modules() if isinstance(m, BlockGate)]
    loss_sum = ce_sum = penalty_sum = 0.0
    correct = seen = steps = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        ce = criterion(outputs, targets)
        penalty = sum(torch.sigmoid(g.logit / g.temperature) for g in gates) / len(gates)
        loss = ce + lambda_sparsity * penalty
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],
                                           grad_clip_norm)
        optimizer.step()
        loss_sum += loss.item()
        ce_sum += ce.item()
        penalty_sum += penalty.item()
        correct += outputs.max(1)[1].eq(targets).sum().item()
        seen += targets.size(0)
        steps += 1
    return {'loss': loss_sum / steps, 'ce': ce_sum / steps, 'penalty': penalty_sum / steps,
            'acc': 100.0 * correct / seen}


def save_ckpt(path, model, epoch, phase, cfg):
    """Weights, gate state and the resolved config (rank 0 only)."""
    if not is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    config = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, type(None), list, tuple))}
    torch.save({'epoch': epoch, 'phase': phase, 'model': raw.state_dict(),
                'gate_state': gate_state_dict(raw), 'config': config}, path)


def run_search(model, loader, test_loader, sampler, optimizer, cfg, local_rank):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']).to(local_rank)
    times = []
    for epoch in range(cfg['epochs']):
        sampler.set_epoch(epoch)
        lr = lr_at(epoch, cfg['base_lr'], cfg['epochs'], cfg['warmup_epochs'])
        set_lr(optimizer, lr)
        model.module.set_temperature(get_current_temp(epoch, cfg['epochs'], cfg['start_temp'], cfg['end_temp']))
        t0 = time.time()
        m = train_epoch(model, loader, criterion, optimizer, local_rank, cfg['lambda_sparsity'], cfg['grad_clip_norm'])
        times.append(time.time() - t0)
        if is_main():
            _, acc = validate(model.module, test_loader, criterion, local_rank)
            logging.info(f"[P1 {epoch + 1}/{cfg['epochs']}] {times[-1]:.1f}s lr={lr:.4f} CE={m['ce']:.3f} "
                         f"penalty={m['penalty']:.4f} trainAcc={m['acc']:.2f}% valAcc={acc:.2f}%")
        dist.barrier()
    return sum(times) / len(times)


def run_threshold(model):
    """Remove every block whose gate is below tau at the current (final) temperature, set the
    other gates to 1 and freeze all gates."""
    raw = model.module
    blocks = gated_blocks(raw)
    removed = 0
    for b in blocks:
        b.snapped = True
        closed = b.gate.value() < TAU
        b.gate.logit.data.fill_(-1000.0 if closed else 1000.0)
        b.gate.logit.requires_grad = False
        removed += int(closed)
    if dist.is_initialized():  # every rank continues from the weights and gates of rank 0
        sd = raw.state_dict()
        for key in sd:
            dist.broadcast(sd[key], src=0)
        raw.load_state_dict(sd)
    return removed, len(blocks)


def run_finetune(model, loader, test_loader, sampler, cfg, local_rank):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']).to(local_rank)
    params = [p for n, p in model.module.named_parameters() if p.requires_grad and 'gate' not in n]
    optimizer = optim.SGD(params, lr=cfg['finetune_lr'], momentum=0.9, weight_decay=1e-3)
    times = []
    for epoch in range(cfg['finetune_epochs']):
        sampler.set_epoch(cfg['epochs'] + epoch)
        set_lr(optimizer, lr_at(epoch, cfg['finetune_lr'], cfg['finetune_epochs'], cfg['finetune_warmup']))
        t0 = time.time()
        m = train_epoch(model, loader, criterion, optimizer, local_rank, 0.0, cfg['grad_clip_norm'])
        times.append(time.time() - t0)
        if is_main():
            _, acc = validate(model.module, test_loader, criterion, local_rank)
            logging.info(f"[P3 {epoch + 1}/{cfg['finetune_epochs']}] {times[-1]:.1f}s loss={m['loss']:.3f} "
                         f"valAcc={acc:.2f}%")
        dist.barrier()
    save_ckpt(os.path.join(cfg['ckpt_dir'], 'final.ckpt'), model, cfg['finetune_epochs'] - 1, 3, cfg)
    return sum(times) / len(times) if times else 0.0


def main():
    cfg = resolve_config(CONFIG)
    local_rank = setup_ddp(cfg['nccl_timeout_min'])
    rank = dist.get_rank()
    setup_logging(rank, cfg['log_dir'])
    set_seed(cfg['seed'], rank)
    try:
        if is_main():
            logging.info(f"World={world_size()} GPUs | model={cfg['model']} dataset={cfg['dataset']} "
                         f"| SyncBN={cfg['use_sync_bn']}")
            logging.info(f"per-GPU batch={cfg['per_gpu_batch']} effective={cfg['per_gpu_batch'] * world_size()} "
                         f"| base_lr={cfg['base_lr']} warmup={cfg['warmup_epochs']} lambda={cfg['lambda_sparsity']} "
                         f"epochs={cfg['epochs']}+{cfg['finetune_epochs']}")
        train_loader, test_loader, sampler, n_train, n_test = build_data(cfg)
        if is_main():
            logging.info(f"train={n_train} test={n_test} classes={cfg['num_classes']}")
        model = build_model(cfg, local_rank)
        optimizer = search_optimizer(model, cfg['base_lr'])

        if is_main():
            logging.info(f"=== PHASE 1: search ({cfg['epochs']} epochs) ===")
        t_search = run_search(model, train_loader, test_loader, sampler, optimizer, cfg, local_rank)

        if is_main():
            logging.info('=== PHASE 2: threshold ===')
        removed, total = run_threshold(model)
        dist.barrier()
        if is_main():
            logging.info(f'Blocks removed: {removed}/{total}')
            logging.info(f"=== PHASE 3: fine-tune ({cfg['finetune_epochs']} epochs) ===")
        t_finetune = run_finetune(model, train_loader, test_loader, sampler, cfg, local_rank)
        if is_main():
            logging.info(f'Mean epoch time: search {t_search:.1f}s, fine-tuning {t_finetune:.1f}s')
    except Exception:
        logging.exception('training failed on this rank')
        raise
    finally:
        cleanup_ddp()
