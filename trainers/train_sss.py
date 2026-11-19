"""The SSS baseline of Table 2 under the LAGD-Net recipe: same data pipeline, seeding, optimizer
for the weights, label smoothing, gradient clipping and epoch budget, with one cosine schedule over
all epochs. The block factors follow the SSS update (lagdnet/sss.py); there is no threshold and no
fine-tuning. At the end, blocks whose factor is exactly zero are removed.

    torchrun --nproc_per_node=1 trainers/train_sss.py --dataset cifar10 --model resnet101 \
        --gamma 0.0106061 --epochs 65 --seed 42 --run_dir results/sss/sss_g0.35_seed42

Rank 0 writes result.json to --run_dir. See scripts/run_sss.sh for the runs of the paper.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import lagdnet.engine as engine  # noqa: E402
from lagdnet import sss  # noqa: E402
from lagdnet.utils import gated_blocks, validate  # noqa: E402


CONFIG_KEYS = ['dataset', 'model', 'epochs', 'per_gpu_batch', 'base_lr', 'warmup_epochs', 'label_smoothing',
               'grad_clip_norm', 'seed', 'use_sync_bn', 'world_size', 'torch_version', 'cuda_version']


@torch.no_grad()
def predictions(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in loader])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'tiny_imagenet'])
    ap.add_argument('--model', default='resnet101')
    ap.add_argument('--gamma', type=float, required=True, help='weight of the L1 penalty on the factors')
    ap.add_argument('--mu', type=float, default=0.9)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    engine.CONFIG.update(dataset=args.dataset, model=args.model, lambda_sparsity=0.0, epochs=args.epochs,
                         finetune_epochs=0, seed=args.seed, use_sync_bn=world > 1,
                         log_dir=args.run_dir, ckpt_dir=os.path.join(args.run_dir, 'ckpts'))
    cfg = engine.resolve_config(engine.CONFIG)
    cfg.update(arm='sss', sss_gamma=args.gamma, sss_mu=args.mu)
    local_rank = engine.setup_ddp(cfg['nccl_timeout_min'])
    rank = dist.get_rank()
    engine.setup_logging(rank, args.run_dir)
    engine.set_seed(cfg['seed'], rank)

    try:
        train_loader, test_loader, sampler, n_train, n_test = engine.build_data(cfg)
        net = engine.MODEL_FACTORIES[args.model](dataset=cfg['model_dataset'], num_classes=cfg['num_classes'])
        net = sss.use_sss_factors(net).to(local_rank)
        if cfg['use_sync_bn']:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        model = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
        blocks = gated_blocks(model.module)
        factors = [b.gate.logit for b in blocks]
        weights = [p for n, p in model.module.named_parameters() if 'gate' not in n]
        optimizer = torch.optim.SGD(weights, lr=cfg['base_lr'], momentum=0.9, weight_decay=1e-3)
        velocities = [torch.zeros_like(f) for f in factors]
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']).to(local_rank)

        if engine.is_main():
            logging.info(f"SSS | {args.model} {args.dataset} gamma={args.gamma} mu={args.mu} seed={args.seed} "
                         f"| GPUs={world} per-GPU batch={cfg['per_gpu_batch']} | epochs={args.epochs} "
                         f"| lr {cfg['base_lr']} cosine, warmup {cfg['warmup_epochs']} | train={n_train} test={n_test}")

        times = []
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            lr = engine.lr_at(epoch, cfg['base_lr'], args.epochs, cfg['warmup_epochs'])
            engine.set_lr(optimizer, lr)
            model.train()
            t0 = time.time()
            loss_sum, correct, seen, steps = 0.0, 0, 0, 0
            for x, y in train_loader:
                x = x.to(local_rank, non_blocking=True)
                y = y.to(local_rank, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                for f in factors:
                    f.grad = None
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],
                                               cfg['grad_clip_norm'])
                optimizer.step()
                sss.apg_step(factors, velocities, lr, args.gamma, args.mu)
                loss_sum += loss.item()
                correct += out.argmax(1).eq(y).sum().item()
                seen += y.size(0)
                steps += 1
            times.append(time.time() - t0)
            if engine.is_main():
                values = [f.item() for f in factors]
                _, acc = validate(model.module, test_loader, criterion, local_rank)
                logging.info(f"[SSS {epoch + 1}/{args.epochs}] {times[-1]:.1f}s lr={lr:.4f} "
                             f"loss={loss_sum / steps:.3f} trainAcc={100 * correct / seen:.2f}% testAcc={acc:.2f}% "
                             f"zero_factors={sum(v == 0.0 for v in values)}/{len(values)}")
            dist.barrier()

        if engine.is_main():
            values = [f.item() for f in factors]
            removed = [i for i, v in enumerate(values) if v == 0.0]
            before = predictions(model.module, test_loader, local_rank)
            _, acc_trained = validate(model.module, test_loader, criterion, local_rank)
            for i in removed:
                sss.remove_block(blocks[i])
            after = predictions(model.module, test_loader, local_rank)
            _, acc_removed = validate(model.module, test_loader, criterion, local_rank)
            same = bool(torch.equal(before, after))
            logging.info(f'[final] test acc {acc_trained:.2f} as trained, {acc_removed:.2f} with the '
                         f'{len(removed)} zero-factor blocks {removed} removed; same predictions: {same}')
            engine.save_ckpt(os.path.join(cfg['ckpt_dir'], 'final.ckpt'), model, args.epochs - 1, 3, cfg)
            cfg.update(world_size=world, torch_version=torch.__version__, cuda_version=torch.version.cuda)
            result = {'arm': 'sss', 'dataset': args.dataset, 'model': args.model, 'seed': args.seed,
                      'gamma': args.gamma, 'mu': args.mu, 'epochs': args.epochs,
                      'blocks_removed': len(removed), 'total_blocks': len(values), 'removed_blocks': removed,
                      'acc': {'as_trained': {'test': round(acc_trained, 3)},
                              'post_removal': {'test': round(acc_removed, 3)}},
                      'removal_changes_predictions': not same,
                      'train_time_h': round(sum(times) / 3600, 2),
                      'config': {k: cfg[k] for k in CONFIG_KEYS}}
            with open(os.path.join(args.run_dir, 'result.json'), 'w') as f:
                json.dump(result, f, indent=2)
        dist.barrier()
    finally:
        engine.cleanup_ddp()


if __name__ == '__main__':
    main()
