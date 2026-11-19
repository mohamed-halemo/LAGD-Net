"""Export benchmark of Sec. 4 ("Measurement").

For each run directory: load ckpts/final.ckpt and build the compact export (removed blocks replaced
by their shortcuts, retained gates or SSS factors folded away). The export must reproduce every
prediction of the trained network on the full test set before it is timed at batch size 1: 20
warm-up and 500 timed passes with CUDA synchronization, median of three measurements. MACs are
counted by thop on the export. A speedup is the latency of the dense export over that of the pruned one.

    python benchmark/benchmark.py --runs <run dirs> --out results/benchmark/benchmark.json
"""

import argparse
import copy
import json
import os
import statistics
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import lagdnet.engine as engine  # noqa: E402
from lagdnet import export, sss  # noqa: E402
from lagdnet.utils import gated_blocks, latency_and_macs, load_gate_state  # noqa: E402

def load_run(run_dir, device):
    """The trained network of a run, its config, result.json and arm (dense, lagnet or sss)."""
    with open(os.path.join(run_dir, 'result.json')) as f:
        res = json.load(f)
    ckpt = torch.load(os.path.join(run_dir, 'ckpts', 'final.ckpt'), map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    arm = res.get('arm') or ('dense' if float(res['config'].get('lambda_sparsity') or 0) == 0 else 'lagnet')
    model = engine.MODEL_FACTORIES[cfg['model']](dataset=cfg['model_dataset'], num_classes=cfg['num_classes'])
    if arm == 'sss':
        sss.use_sss_factors(model)
    model.load_state_dict(ckpt['model'])
    load_gate_state(model, ckpt['gate_state'])
    if arm == 'sss':  # as trained: every block is computed, removed blocks have a factor of exactly 0
        for b in gated_blocks(model):
            b.snapped = False
    return model.to(device).eval(), cfg, res, arm


def compact_copy(model, arm, closed):
    """A gate-free copy of the network without the closed blocks."""
    m = copy.deepcopy(model)
    blocks = gated_blocks(m)
    if arm == 'sss':
        for i in closed:
            sss.remove_block(blocks[i])
    # A retained gate or factor that is not exactly 1 scales the output of the last BatchNorm of
    # its branch, so it is folded into that BatchNorm.
    for i, b in enumerate(blocks):
        v = b.gate.value()
        if i in closed or abs(v - 1.0) <= 1e-6:
            continue
        assert arm != 'dense', f'a dense run has a gate at {v}'
        bn = [mod for name, mod in b.named_modules()
              if isinstance(mod, nn.BatchNorm2d) and not name.startswith('shortcut')][-1]
        with torch.no_grad():
            bn.weight.mul_(v)
            bn.bias.mul_(v)
    export.apply_removal(m, closed)
    for b in gated_blocks(m):
        b.gate = nn.Identity()
        b.snapped = False
    return m


def test_loader(cfg):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg['norm_mean'], cfg['norm_std'])])
    if cfg['dataset'] == 'tiny_imagenet':
        ds = datasets.ImageFolder(os.path.join(cfg['data_root'], 'val'), transform=tf)
    else:
        ds = datasets.CIFAR10(cfg['data_root'], train=False, transform=tf)
    return DataLoader(ds, batch_size=250, shuffle=False, num_workers=4)


@torch.no_grad()
def logits(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).float().cpu() for x, _ in loader])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='run directories with ckpts/final.ckpt and result.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    os.chdir(ROOT)
    device = torch.device(args.device)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}

    for run_dir in args.runs:
        name = os.path.basename(run_dir.rstrip('/'))
        model, cfg, res, arm = load_run(run_dir, device)
        loader = test_loader(cfg)
        closed = sorted(res['removed_blocks']) if arm == 'sss' else export.closed_indices(model)
        compact = compact_copy(model, arm, closed)
        y_model, y_compact = logits(model, loader, device), logits(compact, loader, device)
        agreement = float((y_model.argmax(1) == y_compact.argmax(1)).float().mean())
        assert agreement == 1.0, f'{name}: the export does not reproduce the network (agreement {agreement})'
        latencies = []
        for _ in range(args.repeats):
            ms, gmacs = latency_and_macs(compact, device, tuple(cfg['input_shape']))
            latencies.append(ms)
        rec = {'arm': arm, 'seed': res.get('seed', res['config'].get('seed')), 'removed': closed, 'k': len(closed),
               'total': len(gated_blocks(model)), 'agreement': agreement, 'gmacs': gmacs,
               'latency_ms': round(statistics.median(latencies), 3), 'latency_ms_runs': [round(x, 3) for x in latencies]}
        out[name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k != 'removed'}), flush=True)
        del model, compact
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
