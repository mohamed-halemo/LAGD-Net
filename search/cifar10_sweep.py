"""CIFAR-10 lambda sweep (Sec. 4, "Operating points") and seeds 43 and 44 of the ResNet-101
operating point.

Each ResNet is trained with five to nine values of lambda (seed 42, one GPU, batch 128, or 64 for
ResNet-152, 50 search and 15 fine-tuning epochs); lambda = 0 is the dense baseline.
pick_operating_point() keeps the value that removes the most blocks while staying within 3 points
of dense accuracy (ties: the smaller lambda). The other seeds of the operating points and of the
dense baselines are run by scripts/run_cifar10_seeds.sh.

    GPUS=0,1,2 python search/cifar10_sweep.py               # the sweep, one job per GPU
    GPUS=0,1,2 python search/cifar10_sweep.py --seeds       # seeds 43 and 44 of ResNet-101
    python search/cifar10_sweep.py --plan-only              # print the commands only

Runs that already have a result.json are skipped.
"""

import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'results', 'cifar10_sweep')
GPUS = os.environ.get('GPUS', '0,1,2').split(',')
EPOCHS, FT_EPOCHS = 50, 15
BATCH = {'resnet152': 64}  # 128 for the others
MAX_LOSS_VS_DENSE = 3.0

GRID = {
    'resnet18': [0.0, 0.03, 0.06, 0.09, 0.10, 0.11, 0.12, 0.18, 0.25],
    'resnet34': [0.0, 0.04, 0.08, 0.13, 0.20, 0.30],
    'resnet50': [0.0, 0.03, 0.06, 0.10, 0.11, 0.125, 0.18, 0.25],
    'resnet101': [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    'resnet152': [0.0, 0.10, 0.20, 0.35, 0.45],
}
SEED_MODELS, EXTRA_SEEDS = ['resnet101'], [43, 44]


def run_name(model, lam):
    return f'{model}_cifar10_lam{lam:g}'


def is_done(run_dir):
    try:
        return 'error' not in json.load(open(os.path.join(run_dir, 'result.json')))
    except (OSError, json.JSONDecodeError):
        return False


def command(model, lam, run_dir, port, seed=None):
    cmd = ['torchrun', '--nproc_per_node=1', f'--master_port={port}', 'trainers/train_resnet.py',
           '--model', model, '--dataset', 'cifar10', '--lambda_sparsity', str(lam),
           '--epochs', str(EPOCHS), '--finetune_epochs', str(FT_EPOCHS),
           '--per_gpu_batch', str(BATCH.get(model, 128)), '--run_dir', run_dir]
    return cmd + (['--seed', str(seed)] if seed is not None else [])


def run_pool(jobs, plan_only):
    """jobs: (name, command, run_dir). One job per GPU at a time."""
    jobs = [j for j in jobs if not is_done(j[2])]
    if plan_only:
        for name, cmd, _ in jobs:
            print(name, '::', ' '.join(cmd))
        return
    active = {}
    while jobs or active:
        for gpu in list(active):
            name, proc = active[gpu]
            if proc.poll() is not None:
                print(f'{name} on GPU {gpu} finished (exit {proc.returncode})', flush=True)
                del active[gpu]
        for gpu in [g for g in GPUS if g not in active]:
            if not jobs:
                break
            name, cmd, run_dir = jobs.pop(0)
            os.makedirs(run_dir, exist_ok=True)
            log = open(os.path.join(run_dir, 'driver.log'), 'a')
            log.write(f'\n=== {datetime.now().isoformat()} GPU {gpu}\n{" ".join(cmd)}\n')
            log.flush()
            print(f'{name} -> GPU {gpu}', flush=True)
            active[gpu] = (name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log,
                                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
        time.sleep(20)


def sweep_runs(model):
    """lambda -> result.json of the seed-42 sweep runs of a model."""
    out = {}
    for d in glob.glob(os.path.join(OUT_DIR, f'{model}_cifar10_lam*')):
        if '_seed' in os.path.basename(d) or not is_done(d):
            continue
        r = json.load(open(os.path.join(d, 'result.json')))
        out[float(r['config']['lambda_sparsity'])] = r
    return out


def pick_operating_point(model):
    """The lambda that removes the most blocks within 3 points of the dense run (ties: smaller lambda)."""
    runs = sweep_runs(model)
    dense = runs[0.0]['phase3_val_acc']
    best = None
    for lam, r in sorted(runs.items()):
        if lam == 0.0 or not r.get('blocks_removed'):
            continue
        if dense - r['phase3_val_acc'] > MAX_LOSS_VS_DENSE:
            continue
        key = (r['blocks_removed'], -lam)
        if best is None or key > best[0]:
            best = (key, lam, r)
    assert best, f'no sweep run of {model} qualifies'
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', action='store_true', help='seeds 43 and 44 of the ResNet-101 operating point')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    jobs = []
    if args.seeds:
        for m in SEED_MODELS:
            lam, _ = pick_operating_point(m)
            for i, seed in enumerate(EXTRA_SEEDS):
                run_dir = os.path.join(OUT_DIR, f'{run_name(m, lam)}_seed{seed}')
                jobs.append((os.path.basename(run_dir), command(m, lam, run_dir, 30141 + i, seed), run_dir))
    else:
        for m, lams in GRID.items():
            for lam in lams:
                run_dir = os.path.join(OUT_DIR, run_name(m, lam))
                jobs.append((run_name(m, lam), command(m, lam, run_dir, 30100 + len(jobs)), run_dir))
    run_pool(jobs, args.plan_only)


if __name__ == '__main__':
    main()
