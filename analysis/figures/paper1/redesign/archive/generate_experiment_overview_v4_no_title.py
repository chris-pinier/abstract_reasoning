from pathlib import Path
import os, json
os.environ['MPLCONFIGDIR'] = '/tmp/elife-mpl-cache'
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
B = Path(__file__).resolve().parent
O = B / 'experiment_overview'
O.mkdir(exist_ok=True)
colors = json.loads((B / 'model_palette.json').read_text())
beh = pd.read_csv(B / 'data/behavior_profile_comparison.csv', index_col=0)
rsa = pd.read_csv(B / 'data/matched_rsa.csv')
layers = pd.read_csv(B / 'data/layers.csv')
peaks = pd.read_csv(B / 'data/selected_layers.csv')
regions = ['frontal', 'temporal', 'central', 'occipital', 'parietal']
conditions = ['frp', 'response', 'rest']
cc = {'frp': '#0072B2', 'response': '#D55E00', 'rest': '#009E73'}
ink = '#20303b'
muted = '#607480'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#abb5bd', 'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': muted, 'ytick.color': muted, 'pdf.fonttype': 42})
f = plt.figure(figsize=(14, 11.5), facecolor='white')

def label(letter, title, x, y):
    f.text(x, y, letter, fontsize=16, fontweight='bold', color='#0072B2', va='top')
    f.text(x + 0.027, y - 0.001, title, fontsize=13, fontweight='bold', va='top')

def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.025', facecolor='#f3f6f8', edgecolor='#dce4e9', lw=1))
    ax.text(x + 0.025, y + h - 0.07, title, fontsize=11, fontweight='bold', color=color, va='top')
    ax.text(x + 0.025, y + h - 0.2, body, fontsize=9, color=ink, va='top', linespacing=1.15)

def arrow(ax, p, q, color='#8296a3'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, lw=1.4, color=color, connectionstyle='arc3,rad=0'))
label('A', 'From sequence fixations to representational similarity', 0.04, 0.897)
im = f.add_axes([0.04, 0.736, 0.225, 0.137])
im.imshow(Image.open(B.parents[1] / 'Versions/eLife/figures/trial_example2.png'))
im.axis('off')
from matplotlib.patches import Rectangle
flow = f.add_axes([0, 0, 1, 1], zorder=-1)
flow.set(xlim=(0, 1), ylim=(0, 1))
flow.axis('off')
starts = [0.1, 0.39, 0.67]
cols = ['#0072B2', '#008778', '#69a8d0']
t = np.linspace(0, 1.3, 700)
y = 0.2 * np.sin(2 * np.pi * 7 * t) + 0.13 * np.sin(2 * np.pi * 13 * t + 0.7) + 0.1 * np.cos(2 * np.pi * 3 * t)
xs = [0.3, 0.452, 0.605]
w = 0.12
for x, title in zip(xs, ['1  Mark fixations', '2  Align epochs', '3  Average within trial']):
    f.text(x, 0.872, title, fontsize=9, fontweight='bold')

def trace_axes(x):
    ax = f.add_axes([x, 0.775, w, 0.075])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.tick_params(axis='x', labelsize=7, length=2)
    return ax
ax = trace_axes(xs[0])
ax.plot(t, y, color=ink, lw=0.8)
for i, (st, col) in enumerate(zip(starts, cols)):
    ax.add_patch(Rectangle((st, 0.53 + i * 0.18), 0.6, 0.12, color=col, alpha=0.28, lw=0))
    ax.axvline(st, color=col, ls=':', lw=0.8)
ax.set(xlim=(0, 1.3), ylim=(-0.55, 1.12), xticks=[0, 0.6, 1.2])
ax.set_xlabel('Time in trial (s)', fontsize=7, labelpad=2)
epoch_t = np.linspace(0, 0.6, 220)
epochs = np.array([np.interp(st + epoch_t, t, y) for st in starts])
ax = trace_axes(xs[1])
for i, (ep, col) in enumerate(zip(epochs, cols)):
    ax.plot(epoch_t * 1000, ep + (2 - i) * 0.65, color=col, lw=1)
ax.axvline(0, color=muted, lw=0.7)
ax.set(xlim=(-10, 610), xticks=[0, 600])
ax.set_xlabel('From fixation onset (ms)', fontsize=7, labelpad=2)
ax = trace_axes(xs[2])
for ep in epochs:
    ax.plot(epoch_t * 1000, ep, color='#adcde0', lw=0.65, alpha=0.8)
ax.plot(epoch_t * 1000, epochs.mean(axis=0), color=cc['frp'], lw=2)
ax.set(xlim=(0, 600), xticks=[0, 600], ylim=(-0.6, 0.6))
ax.set_xlabel('From fixation onset (ms)', fontsize=7, labelpad=2)
for p, q in [((0.269, 0.815), (0.293, 0.815)), ((0.425, 0.815), (0.446, 0.815)), ((0.578, 0.815), (0.599, 0.815))]:
    arrow(flow, p, q)
features = np.array([[np.sin(k * 0.65 + j * 0.6) for j in range(12)] for k in range(8)])
rdms = [1 - np.corrcoef(features), 1 - np.corrcoef(features + 0.3 * np.cos(np.arange(8)[:, None] + np.arange(12)[None, :]))]
for yy, title, rdm in zip([0.811, 0.735], ['Human group RDM', 'LLM pattern RDM'], rdms):
    f.text(0.762, yy + 0.054, title, fontsize=8, fontweight='bold')
    mat = f.add_axes([0.766, yy, 0.05, 0.05])
    mat.imshow(rdm, cmap='Blues', vmin=0, vmax=2)
    mat.axis('off')
    arrow(flow, (0.822, yy + 0.025), (0.858, 0.799))
flow.add_patch(FancyBboxPatch((0.862, 0.771), 0.092, 0.057, boxstyle='round,pad=0.004', facecolor='#f3f6f8', edgecolor='#dce4e9'))
f.text(0.908, 0.81, 'RSA', ha='center', fontsize=11, fontweight='bold', color=cc['frp'])
f.text(0.908, 0.785, 'Pearson r', ha='center', fontsize=8)
f.text(0.96, 0.717, '28 matched pattern pairs', ha='right', fontsize=8, color=muted)
arrow(flow, (0.726, 0.812), (0.76, 0.836))
f.lines.append(Line2D([0.04, 0.96], [0.69, 0.69], transform=f.transFigure, color='#dce4e9', lw=1))
label('B', 'Accuracy across abstract patterns', 0.04, 0.669)
order = beh.sort_values('accuracy', ascending=False).index.tolist()
mp = pd.read_csv(B / 'data/model_pattern_accuracy.csv', index_col=0)
hp = pd.read_csv(B / 'data/human_pattern_summary.csv').set_index('pattern').reindex(mp.index)
ax = f.add_axes([0.087, 0.41, 0.417, 0.17])
x = np.arange(len(mp))
for m in order:
    ax.plot(x, 100 * mp[m], color=colors[m], marker='o', markersize=3, lw=1.5, label=m)
ax.fill_between(x, 100 * hp['ci_low'], 100 * hp['ci_high'], color=ink, alpha=0.15, lw=0, zorder=1)
ax.plot(x, 100 * hp['mean'], color=ink, marker='D', markersize=4, lw=2.4, label='Humans', zorder=6)
ax.axhline(25, color='#9aa8b0', ls=':', lw=1)
ax.set(ylim=(-3, 105), xlim=(-0.2, 7.2), yticks=[0, 25, 50, 75, 100], ylabel='Accuracy (%)', xticks=x, xticklabels=mp.index)
ax.tick_params(axis='x', labelsize=8, rotation=35)
for tick in ax.get_xticklabels():
    tick.set_ha('right')
ax.grid(axis='y', color='#edf0f3', lw=0.7)
h, l = ax.get_legend_handles_labels()
legend_order = ['Humans', 'Phi-4', 'Qwen2.5-72B', 'Gemma-2-2B', 'Gemma-2-9B', 'Gemma-2-27B', 'Llama-3.2-3B', 'Llama-3.3-70B', 'DeepSeek-R1-70B']
legend_handles = dict(zip(l, h))
f.legend([legend_handles[m] for m in legend_order], legend_order, loc='upper left', bbox_to_anchor=(0.062, 0.643), ncol=3, frameon=False, fontsize=8.3, columnspacing=1.2, handlelength=1.8, labelspacing=0.55)
label('C', 'Task structure emerges inside the models', 0.55, 0.669)
ax = f.add_axes([0.585, 0.389, 0.375, 0.222])
for m in order:
    g = layers[layers.model.eq(m)].sort_values('layer')
    x = g.layer / g.layer.max()
    ax.plot(x, g.similarity, color=colors[m], lw=1.6, alpha=0.95)
    p = peaks[peaks.model.eq(m)].iloc[0]
    ax.scatter(p.normalized_depth, p.r, s=32, color=colors[m], edgecolors='white', lw=0.7, zorder=4)
ax.set(xlim=(0, 1), ylim=(-0.03, 0.83), xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8], xlabel='Relative layer depth', ylabel='Task-design RSA (r)')
ax.grid(axis='y', color='#edf0f3', lw=0.7)
f.lines.append(Line2D([0.04, 0.96], [0.321, 0.321], transform=f.transFigure, color='#dce4e9', lw=1))
label('D', 'Frontal correspondence is strongest for FRP', 0.04, 0.304)
ax = f.add_axes([0.095, 0.103, 0.36, 0.155])
means = rsa.groupby(['channel_group', 'eeg_type']).rsa_similarity.mean().unstack().reindex(index=regions, columns=conditions)
for j, c in enumerate(conditions):
    vals = rsa[rsa.channel_group.eq('frontal') & rsa.eeg_type.eq(c)].set_index('label').reindex(order).rsa_similarity.to_numpy()
    ax.scatter(j + np.linspace(-0.14, 0.14, 8), vals, color=cc[c], s=28, alpha=0.55, zorder=3)
    ax.plot([j - 0.2, j + 0.2], [vals.mean()] * 2, color=cc[c], lw=3, zorder=4)
    ax.text(j, vals.max() + 0.05, f'{vals.mean():.2f}', ha='center', fontsize=15, color=cc[c], fontweight='bold')
ax.axhline(0, color='#8a969f', lw=0.7)
ax.set(xlim=(-0.5, 2.5), ylim=(-0.09, 0.83), xticks=[0, 1, 2], xticklabels=['FRP', 'Response', 'Rest'], ylabel='LLM-human RSA (r)', yticks=[0, 0.2, 0.4, 0.6, 0.8])
for t, c in zip(ax.get_xticklabels(), conditions):
    t.set_color(cc[c])
    t.set_fontweight('bold')
ax.grid(axis='y', color='#edf0f3', lw=0.7)
label('E', 'Regional context matters', 0.55, 0.304)
ax = f.add_axes([0.595, 0.105, 0.365, 0.145])
values = np.vstack([means.values, means.mean().values])
x = np.array([0, 1, 2, 3, 4, 5.5])
width = 0.23
ax.axvspan(4.9, 6.08, color='#f2f5f7', zorder=0)
for j, c in enumerate(conditions):
    bars = ax.bar(x + (j - 1) * width, values[:, j], width=width * 0.9, color=cc[c], label='FRP' if c == 'frp' else c.title(), zorder=3)
    for bar, value in zip(bars, values[:, j]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.02 if value >= 0 else -0.025), f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=6.5, color=cc[c])
ax.axhline(0, color='#697780', lw=0.9)
ax.set(xticks=x, xticklabels=['Frontal', 'Temporal', 'Central', 'Occipital', 'Parietal', 'Regional\nmean'], ylim=(-0.3, 0.75), yticks=[-0.2, 0, 0.2, 0.4, 0.6], ylabel='Mean LLM-human RSA (r)', xlim=(-0.6, 6.1))
ax.tick_params(axis='x', labelsize=7.5)
ax.tick_params(axis='y', labelsize=8)
ax.yaxis.label.set_size(9)
ax.grid(axis='y', color='#e5eaee', lw=0.7)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.03), ncol=3, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.5)
out = O / 'main_experiment_overview_v4_frp_construction_no_title'
from matplotlib.text import Text
for item in f.findobj(match=Text):
    item.set_text(item.get_text().replace('DeepSeek-R1-70B', 'DS-R1-Llama-70B'))
f.savefig(out.with_suffix('.pdf'), bbox_inches=matplotlib.transforms.Bbox.from_extents(.25,.6,13.7,10.5), dpi=600, facecolor='white')
f.savefig(out.with_suffix('.png'), bbox_inches=matplotlib.transforms.Bbox.from_extents(.25,.6,13.7,10.5), dpi=220, facecolor='white')
plt.close(f)
print(out.with_suffix('.pdf'))
