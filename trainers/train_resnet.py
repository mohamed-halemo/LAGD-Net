"""One LAGD-Net run (search, threshold, fine-tuning). Rank 0 writes result.json to --run_dir.

    torchrun --nproc_per_node=1 trainers/train_resnet.py --model resnet101 --dataset cifar10 \
        --lambda_sparsity 0.2 --epochs 50 --finetune_epochs 15 --per_gpu_batch 128 --seed 42 \
        --run_dir runs/resnet101_cifar10_lam0.2

--lambda_sparsity 0 gives the dense baseline.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

CONFIG_KEYS = ['dataset', 'model', 'lambda_sparsity', 'epochs', 'finetune_epochs', 'per_gpu_batch', 'base_lr',
               'warmup_epochs', 'finetune_lr', 'finetune_warmup', 'label_smoothing', 'grad_clip_norm', 'start_temp',
               'end_temp', 'seed', 'use_sync_bn', 'world_size', 'torch_version', 'cuda_version']


def parse_log(path):
    """Accuracy at the end of the search (soft gates) and after fine-tuning, the blocks removed and
    the training time (sum of the epoch times), read from rank0.log."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    result = {}
    p1 = re.findall(r'\[P1 \d+/\d+\].*?valAcc=([\d.]+)%', text)
    if p1:
        result['phase1_val_acc'] = float(p1[-1])
    removed = re.search(r'Blocks removed: (\d+)/(\d+)', text)
    if removed:
        result['blocks_removed'] = int(removed.group(1))
        result['total_blocks'] = int(removed.group(2))
    p3 = re.findall(r'\[P3 \d+/\d+\].*?valAcc=([\d.]+)%', text)
    if p3:
        result['phase3_val_acc'] = float(p3[-1])
    if 'phase1_val_acc' in result and 'phase3_val_acc' in result:
        result['accuracy_drop'] = round(result['phase1_val_acc'] - result['phase3_val_acc'], 3)
    result['train_time_h'] = round(sum(float(t) for t in re.findall(r'\[P[13] \d+/\d+\] ([\d.]+)s', text)) / 3600, 2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'tiny_imagenet'])
    ap.add_argument('--lambda_sparsity', type=float, required=True)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--finetune_epochs', type=int, default=None)
    ap.add_argument('--per_gpu_batch', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()

    import torch
    import lagdnet.engine as engine

    os.makedirs(args.run_dir, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    engine.CONFIG.update(dataset=args.dataset, model=args.model, lambda_sparsity=args.lambda_sparsity,
                         epochs=args.epochs, finetune_epochs=args.finetune_epochs,
                         per_gpu_batch=args.per_gpu_batch, seed=args.seed, use_sync_bn=world > 1,
                         log_dir=args.run_dir, ckpt_dir=os.path.join(args.run_dir, 'ckpts'))
    engine.main()

    if int(os.environ.get('RANK', '0')) == 0:
        result = parse_log(os.path.join(args.run_dir, 'rank0.log'))
        cfg = dict(engine.resolve_config(engine.CONFIG), world_size=world,
                   torch_version=torch.__version__, cuda_version=torch.version.cuda)
        result['config'] = {k: cfg[k] for k in CONFIG_KEYS}
        with open(os.path.join(args.run_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
