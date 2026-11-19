"""Tables 1 and 2 of the paper from the run records in results/ (no GPU needed).

Operating points follow Sec. 4: on CIFAR-10, the sweep value that removes the most blocks within
3 points of the dense run (search/cifar10_sweep.py), then three seeds of it; on Tiny-ImageNet, the
run with the most blocks removed within 5 points of the soft-gated network at the end of the search
(search/tiny_search.py), seed 42. Speedups and MACs come from the export benchmark
(results/benchmark): the mean latency of the dense exports over that of the pruned exports.

    python scripts/make_paper_tables.py
"""

import glob
import json
import os
import re
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'search'))
from cifar10_sweep import pick_operating_point  # noqa: E402
from tiny_search import operating_point  # noqa: E402

MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
SW, SE = 'results/cifar10_sweep', 'results/cifar10_seeds'
TINY_DENSE = {m: f'results/tiny_dense_matched/{m}_tiny_dense' for m in ('resnet18', 'resnet34', 'resnet50', 'resnet152')}
TINY_DENSE['resnet101'] = 'results/tiny_dense_r101/dense_seed42'


def rj(path):
    with open(path) as f:
        return json.load(f)


BENCH = rj('results/benchmark/benchmark.json')
BENCH.update(rj('results/benchmark/benchmark_resnet50_lam0.11.json'))  # pruned ResNet-50 exports, timed in a second session
mean = st.mean


def latencies():
    """Mean latency of the dense and of the pruned exports for each dataset and depth."""
    groups = {}
    for name, r in BENCH.items():
        ds = 'tiny' if ('tiny_imagenet' in name or 'attempt' in name) else 'cifar10'
        depth = re.search(r'resnet(\d+)', name).group(1)
        g = groups.setdefault(f'{ds}_resnet{depth}', {'dense': [], 'pruned': []})
        g['dense' if r['arm'] == 'dense' else 'pruned'].append(r['latency_ms'])
    return {k: {'dense_ms': mean(v['dense']), 'speedup': mean(v['dense']) / mean(v['pruned'])} for k, v in groups.items()}


LATENCY = latencies()


def r2(x):
    return round(x, 2)


def d_dense(post, dense):
    return round(r2(mean(post)) - r2(mean(dense)), 2)  # difference of the printed means


def lam_str(x, places):
    if Decimal(str(x)) == Decimal(str(x)).quantize(Decimal('0.01')):  # 0.40 rather than 0.400
        places = '0.01'
    return str(Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def macs_cut(run, dense):
    """MAC reduction (%) of the export of a run against the dense export."""
    return 100 * (1 - BENCH[run]['gmacs'] / BENCH[dense]['gmacs'])


def c10_seed_dirs(model, lam):
    """The three seeds of one CIFAR-10 setting (seed 42 is the sweep run)."""
    tag = '0.0' if lam == 0 else f'{lam:g}'
    extra = [d for s in (43, 44) for d in (f'{SE}/{model}_cifar10_lam{tag}_seed{s}', f'{SW}/{model}_cifar10_lam{tag}_seed{s}')
             if os.path.exists(f'{d}/result.json')]
    assert len(extra) == 2, (model, lam, extra)
    return [f'{SW}/{model}_cifar10_lam{lam:g}'] + extra


def table1():
    print('=== Table 1 ===')
    print('Model | lambda | Removed | Acc. (%) | Delta | Speedup | MACs down')
    print('-- CIFAR-10 (three-seed means) --')
    out = {}
    for m in MODELS:
        lam, _ = pick_operating_point(m)
        dirs = c10_seed_dirs(m, lam)
        R = [rj(f'{d}/result.json') for d in dirs]
        D = [rj(f'{d}/result.json') for d in c10_seed_dirs(m, 0.0)]
        post = [r['phase3_val_acc'] for r in R]
        dense = [r['phase3_val_acc'] for r in D]
        row = dict(lam=lam, removed=mean(r['blocks_removed'] for r in R), total=R[0]['total_blocks'],
                   acc=r2(mean(post)), acc_std=r2(st.stdev(post)), dense=r2(mean(dense)), dense_std=r2(st.stdev(dense)),
                   delta=d_dense(post, dense), speedup=LATENCY[f'cifar10_{m}']['speedup'],
                   macs=mean(macs_cut(os.path.basename(d), f'{m}_cifar10_lam0') for d in dirs))
        out[('cifar10', m)] = row
        print(f"{m} | {lam_str(lam, '0.01')} | {row['removed']:.1f}/{row['total']} | {row['acc']:.2f} | {row['delta']:+.2f} | "
              f"{row['speedup']:.2f}x | {row['macs']:.1f}%")
    print('-- Tiny-ImageNet (seed 42) --')
    worst = 0.0
    for m in MODELS:
        attempt, r = operating_point(m)
        run = f'{m}_attempt{attempt:02d}'
        dense = rj(f'{TINY_DENSE[m]}/result.json')['phase3_val_acc']
        lam = r['config']['lambda_sparsity']
        worst = max(worst, r['accuracy_drop'])
        row = dict(lam=lam, removed=r['blocks_removed'], total=r['total_blocks'], acc=r['phase3_val_acc'],
                   dense=dense, delta=round(r['phase3_val_acc'] - dense, 2),
                   speedup=LATENCY[f'tiny_{m}']['speedup'], macs=macs_cut(run, f'{m}_tiny_imagenet'))
        out[('tiny', m)] = row
        print(f"{m} | {lam_str(lam, '0.001')} | {row['removed']}/{row['total']} | {row['acc']:.2f} | {row['delta']:+.2f} | "
              f"{row['speedup']:.2f}x | {row['macs']:.1f}%   ({run})")
    print(f'Largest loss against the soft-gated network at the end of the search: {worst:.1f} points')
    return out


def table2(t1):
    print('\n=== Table 2: ResNet-101, LAGD-Net and SSS ===')
    print('Method | Removed | Acc. (%) | Delta | Speedup | MACs down')
    c10 = t1[('cifar10', 'resnet101')]
    print(f"-- CIFAR-10, three seeds; dense {c10['dense']:.2f} +- {c10['dense_std']:.2f} --")
    runs = sorted(glob.glob('results/sss/sss_g0.35_seed*/result.json'))
    if runs and os.path.exists('results/sss/bench_g0.35.json'):
        R = [rj(p) for p in runs]
        B = [r for r in rj('results/sss/bench_g0.35.json').values() if r['arm'] == 'sss']
        acc = [r['acc']['as_trained']['test'] for r in R]
        dense_gmacs = BENCH['resnet101_cifar10_lam0']['gmacs']
        print(f"SSS (gamma = 0.35/33) | {mean(r['blocks_removed'] for r in R):.1f}/33 | {mean(acc):.2f} +- {st.stdev(acc):.2f} | "
              f"{round(r2(mean(acc)) - c10['dense'], 2):+.2f} | "
              f"{LATENCY['cifar10_resnet101']['dense_ms'] / mean(b['latency_ms'] for b in B):.2f}x | "
              f"{100 * (1 - round(mean(b['gmacs'] for b in B), 3) / round(dense_gmacs, 3)):.1f}%")
    else:
        print('SSS | no records in results/sss')
    print(f"LAGD-Net (lambda = 0.2) | {c10['removed']:.1f}/33 | {c10['acc']:.2f} +- {c10['acc_std']:.2f} | "
          f"{c10['delta']:+.2f} | {c10['speedup']:.2f}x | {c10['macs']:.1f}%")

    tiny = t1[('tiny', 'resnet101')]
    print(f"-- Tiny-ImageNet, seed 42; dense {tiny['dense']:.2f} --")
    run, joint = 'results/sss/tiny_g0.7_seed42/result.json', 'results/sss/bench_tiny_joint.json'
    if os.path.exists(run) and os.path.exists(joint):
        r = rj(run)
        J = {b['arm']: b for b in rj(joint).values()}
        # timed together with LAGD-Net: scale its Table 1 speedup by their latency ratio
        speedup = tiny['speedup'] * J['lagnet']['latency_ms'] / J['sss']['latency_ms']
        acc = r['acc']['as_trained']['test']
        print(f"SSS (gamma = 0.7/33) | {r['blocks_removed']}/{r['total_blocks']} | {acc:.2f} | "
              f"{round(acc - tiny['dense'], 2):+.2f} | {speedup:.2f}x | "
              f"{100 * (1 - J['sss']['gmacs'] / J['dense']['gmacs']):.1f}%")
    else:
        print('SSS | no records in results/sss')
    print(f"LAGD-Net (lambda = 0.40) | {tiny['removed']}/{tiny['total']} | {tiny['acc']:.2f} | {tiny['delta']:+.2f} | "
          f"{tiny['speedup']:.2f}x | {tiny['macs']:.1f}%")


if __name__ == '__main__':
    table2(table1())
