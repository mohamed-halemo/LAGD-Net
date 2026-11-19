"""Fig. 1: (a) the gated residual block of Eq. 1, (b) the network after the threshold.
Writes figs/lag_net_block.{pdf,png}.

    python figs/make_fig1_block.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

INK, MUT, GRID = '#0b0b0b', '#52514e', '#c9c8c4'
BLUE, LBLUE, GREY = '#2a78d6', '#dbe8fa', '#eeeeec'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})

fig = plt.figure(figsize=(6.2, 1.98), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.06)

def box(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, fs=8.5, tc=INK, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=tc, linespacing=1.15)

def arrow(ax, p, q, color=INK, lw=1.0, style='-|>', cs=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 connectionstyle=cs or 'arc3,rad=0', shrinkA=0, shrinkB=0))

# (a) gated residual block
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.text(0, 5.95, '(a) Gated block, Eq. 1 (search)', fontsize=9, color=INK, va='top')
ax.text(0.35, 2.75, r'$\mathbf{x}_i$', ha='center', va='center', fontsize=9.5)
arrow(ax, (0.75, 2.75), (1.55, 2.75))
box(ax, 1.55, 2.0, 3.55, 1.5, 'residual\n' + r'branch $\mathcal{F}$', fs=8.5)
arrow(ax, (5.1, 2.75), (6.0, 2.75))
ax.add_patch(Circle((6.4, 2.75), 0.4, fc='white', ec=INK, lw=1.0)); ax.text(6.4, 2.75, r'$\times$', ha='center', va='center', fontsize=10)
box(ax, 4.3, 3.95, 4.2, 1.0, r'gate $\sigma(g_i/T)$', fc='#fff4d6', ec='#d99a00', fs=8.5)
arrow(ax, (6.4, 3.95), (6.4, 3.17))
arrow(ax, (6.8, 2.75), (7.75, 2.75))
ax.add_patch(Circle((8.15, 2.75), 0.4, fc='white', ec=INK, lw=1.0)); ax.text(8.15, 2.75, '+', ha='center', va='center', fontsize=11)
arrow(ax, (8.55, 2.75), (9.2, 2.75))
ax.text(9.62, 2.75, r'$\mathbf{y}_i$', ha='center', va='center', fontsize=9.5)
ax.plot([1.15, 1.15], [2.75, 1.05], color=INK, lw=1.0)
ax.plot([1.15, 8.15], [1.05, 1.05], color=INK, lw=1.0)
arrow(ax, (8.15, 1.05), (8.15, 2.33))
ax.text(4.65, 0.45, r'shortcut $\mathcal{S}_i$', ha='center', va='center', fontsize=8.5, color=MUT)

# (b) network after the threshold
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 12.5); ax.set_ylim(0, 6); ax.axis('off')
ax.text(0.2, 5.95, r'(b) Pruned network (inference)', fontsize=9, color=INK, va='top')
xs = [1.0, 3.75, 6.5, 9.25]; w, h, y0 = 2.3, 1.5, 2.0
states = [1, 0, 0, 1]
for k, (x, s) in enumerate(zip(xs, states)):
    if s:
        box(ax, x, y0, w, h, f'block {k+1}\n' + r'gate 1', fs=8)
    else:
        box(ax, x, y0, w, h, f'block {k+1}\n' + r'gate 0', fc=GREY, ec=GRID, fs=8, tc=MUT, ls='--')
ax.text(0.22, y0 + h/2, 'in', ha='center', va='center', fontsize=8.5, color=MUT)
arrow(ax, (0.62, y0 + h/2), (xs[0], y0 + h/2))
for a, b in ((0, 1), (1, 2), (2, 3)):
    arrow(ax, (xs[a] + w, y0 + h/2), (xs[b], y0 + h/2), color=GRID, lw=0.9)
arrow(ax, (xs[3] + w, y0 + h/2), (12.0, y0 + h/2))
ax.text(12.1, y0 + h/2, 'out', ha='left', va='center', fontsize=8.5, color=MUT)
arrow(ax, (xs[0] + w, y0 + h/2 + 0.25), (xs[3], y0 + h/2 + 0.25), color=BLUE, lw=1.5, cs='arc3,rad=-0.42')
ax.text((xs[0] + w + xs[3]) / 2, 4.72, 'shortcuts only', ha='center', va='center', fontsize=8.5, color=BLUE)
ax.text(6.25, 0.45, r'gate $<\tau$: residual branch removed', ha='center', va='center', fontsize=8.5, color=MUT)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lag_net_block')
fig.savefig(out + '.png', facecolor='white', bbox_inches='tight', pad_inches=0.02)
fig.savefig(out + '.pdf', facecolor='white', bbox_inches='tight', pad_inches=0.02)
