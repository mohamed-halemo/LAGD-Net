"""Tiny-ImageNet operating points (Sec. 4, "Operating points").

The lambda values were tried one after another, each chosen from the results before it; RUNS lists
every run in that order. A model's operating point is the run that removes the most blocks while its
final accuracy stays within 5 points of the soft-gated network at the end of the search (ties: the
smaller loss). All runs use seed 42 and three GPUs with synchronized batch normalization.

    GPUS=0,1,2 python search/tiny_search.py      # run what is missing, then print the operating points
    python search/tiny_search.py --pick          # print the operating points only
"""

import argparse
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPUS = os.environ.get('GPUS', '0,1,2')
MAX_LOSS = 5.0
BATCH = {'resnet152': 64}  # per-GPU batch, 128 for the others

# (model, lambda, fine-tuning epochs, attempt)
RUNS = [
    ('resnet18', 0.175, 55, 1), ('resnet18', 0.2188, 55, 2), ('resnet18', 0.16, 65, 3), ('resnet18', 0.1894, 65, 4),
    ('resnet34', 0.213, 55, 1), ('resnet34', 0.2, 65, 2), ('resnet34', 0.2065, 65, 3),
    ('resnet50', 0.185, 55, 1), ('resnet50', 0.2313, 55, 2), ('resnet50', 0.17, 65, 3), ('resnet50', 0.2006, 65, 4),
    ('resnet101', 0.26, 55, 1), ('resnet101', 0.338, 55, 2), ('resnet101', 0.4, 55, 3),
    ('resnet152', 0.32, 55, 1),
    ('resnet101', 0.45, 55, 4), ('resnet152', 0.4, 55, 2),
    ('resnet18', 0.175, 55, 5), ('resnet34', 0.2065, 65, 4), ('resnet50', 0.185, 55, 5),  # repeats of the best so far
    ('resnet34', 0.213, 75, 5), ('resnet50', 0.2, 75, 6), ('resnet101', 0.42, 65, 5), ('resnet152', 0.44, 55, 3),
]


def run_dir(model, attempt):
    group = 'resnet_101_152' if model in ('resnet101', 'resnet152') else 'resnet_18_34_50'
    return os.path.join(ROOT, 'results', group, f'{model}_attempt{attempt:02d}')


def result(model, attempt):
    path = os.path.join(run_dir(model, attempt), 'result.json')
    return json.load(open(path)) if os.path.exists(path) else None


def launch(model, lam, ft_epochs, attempt):
    d = run_dir(model, attempt)
    os.makedirs(d, exist_ok=True)
    cmd = ['torchrun', '--nproc_per_node=3', f'--master_port={29500 + attempt * 10}', 'trainers/train_resnet.py',
           '--model', model, '--dataset', 'tiny_imagenet', '--lambda_sparsity', str(lam),
           '--finetune_epochs', str(ft_epochs), '--per_gpu_batch', str(BATCH.get(model, 128)), '--run_dir', d]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS))


def operating_point(model):
    """The run with the most blocks removed within MAX_LOSS points of the soft-gated network."""
    done = [(r['blocks_removed'], -r['accuracy_drop'], attempt, r) for m, _, _, attempt in RUNS
            if m == model and (r := result(m, attempt)) and 'accuracy_drop' in r]
    ok = [x for x in done if -x[1] <= MAX_LOSS]
    return max(ok)[2:] if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pick', action='store_true', help='only print the operating points')
    args = ap.parse_args()
    if not args.pick:
        for model, lam, ft_epochs, attempt in RUNS:
            if result(model, attempt) is None:
                launch(model, lam, ft_epochs, attempt)
    print('model | lambda | FT epochs | removed | acc. end of search | final acc. | loss | run')
    for model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        pick = operating_point(model)
        if pick is None:
            print(f'{model} | no run within {MAX_LOSS} points')
            continue
        attempt, r = pick
        print(f"{model} | {r['config']['lambda_sparsity']} | {r['config']['finetune_epochs']} | "
              f"{r['blocks_removed']}/{r['total_blocks']} | {r['phase1_val_acc']:.2f} | {r['phase3_val_acc']:.2f} | "
              f"{r['accuracy_drop']:.2f} | {os.path.relpath(run_dir(model, attempt), ROOT)}")


if __name__ == '__main__':
    main()
