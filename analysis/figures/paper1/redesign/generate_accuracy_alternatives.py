"""Standalone Figure 1B alternatives using the same cached empirical inputs.

Run with the figure Python environment. PAPER_FIGURE_OUTPUT overrides the
default variants/figure_1b output directory in this source bundle. No manuscript edits.
Human intervals are the existing pointwise participant-bootstrap 95% CIs;
model accuracies are descriptive values without added uncertainty estimates.
"""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/elife-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

B = Path(__file__).resolve().parent
O = Path(os.environ.get('PAPER_FIGURE_OUTPUT', str(B.parent / 'variants/figure_1b')))
O.mkdir(parents=True, exist_ok=True)
palette = json.loads((B / 'model_palette.json').read_text())
models = ['Gemma-2-2B', 'Gemma-2-9B', 'Gemma-2-27B',
          'Llama-3.2-3B', 'Llama-3.3-70B', 'DeepSeek-R1-70B',
          'Phi-4', 'Qwen2.5-72B']
labels = ['Humans'] + [m.replace('DeepSeek-R1-70B', 'DS-R1-Llama-70B') for m in models]
mp = pd.read_csv(B / 'data/model_pattern_accuracy.csv', index_col=0)
hp = pd.read_csv(B / 'data/human_pattern_summary.csv').set_index('pattern').loc[mp.index]
values = 100 * np.vstack([hp['mean'].to_numpy(), mp[models].to_numpy().T])
errors = 100 * np.vstack([hp['mean'] - hp.ci_low, hp.ci_high - hp['mean']])
assert values.shape == (9, 8) and np.isfinite(values).all()
assert ((values >= 0) & (values <= 100)).all() and (errors >= 0).all()
ink = '#20303b'
colors = [ink] + [palette[m] for m in models]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': ink, 'axes.labelcolor': ink,
                     'xtick.color': ink, 'ytick.color': ink,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b4bec5', 'pdf.fonttype': 42})

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(O / f'{name}.{ext}', dpi=220, facecolor='white')
    plt.close(fig)

# Shared column positions preserve the pattern order used in Figure 1A.
x = np.arange(8)
fig = plt.figure(figsize=(11.5, 7.8))
fig.suptitle('Accuracy across abstract patterns', x=.235, y=.965,
             ha='left', fontsize=17, fontweight='bold')
axh = fig.add_axes([.235, .735, .69, .16])
axh.errorbar(x, values[0], yerr=errors, fmt='D', color=ink,
             markersize=5, capsize=4, elinewidth=1.3)
axh.set(xlim=(-.5, 7.5), ylim=(0, 105), yticks=[0, 25, 50, 75, 100],
        ylabel='Human accuracy (%)')
axh.axhline(25, color='#8796a0', ls=':', lw=1)
axh.text(7.48, 27, 'Chance', ha='right', va='bottom', fontsize=9, color='#657581')
axh.grid(axis='y', color='#e8edf0', lw=.6)
axh.tick_params(axis='x', bottom=False, labelbottom=False)
axh.set_title('Participant mean and 95% bootstrap interval', loc='left', fontsize=10)

ax = fig.add_axes([.235, .205, .69, .49])
# Empty rows separate humans and the model families without changing the scale.
rows = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11])
grid = np.full((12, 8), np.nan)
grid[rows] = values
im = ax.imshow(np.ma.masked_invalid(grid), cmap='Blues', vmin=0, vmax=100, aspect='auto')
for r, v in zip(rows, values):
    for j, value in enumerate(v):
        ax.text(j, r, f'{value:.0f}', ha='center', va='center', fontsize=10,
                color='white' if value >= 62 else ink)
ax.set(xticks=x, xticklabels=mp.index, yticks=rows, yticklabels=labels)
ax.tick_params(axis='both', length=0, pad=8)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
for tick, color in zip(ax.get_yticklabels(), colors):
    tick.set_color(color)
    tick.set_fontweight('bold')
for spine in ax.spines.values():
    spine.set_visible(False)
cax = fig.add_axes([.235, .09, .69, .018])
fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 25, 50, 75, 100], label='Accuracy (%)')
fig.text(.235, .018, 'Cell labels rounded to whole percentages; colour encodes unrounded values. Humans: N = 25.',
         fontsize=9, color='#657581')
save(fig, 'figure_1b_accuracy_heatmap')

# Grouped bars retain the same family/size ordering within every pattern.
fig, ax = plt.subplots(figsize=(15, 6.8))
fig.subplots_adjust(left=.065, right=.985, bottom=.205, top=.75)
fig.suptitle('Accuracy across abstract patterns', x=.065, y=.965,
             ha='left', fontsize=17, fontweight='bold')
width = .088
for i, (label, color) in enumerate(zip(labels, colors)):
    positions = x + (i - 4) * width
    ax.bar(positions, values[i], width=width*.91, color=color, zorder=3)
    if i == 0:
        ax.errorbar(positions, values[i], yerr=errors, fmt='none',
                    ecolor=ink, capsize=2, elinewidth=1, zorder=4)
ax.axhline(25, color='#788892', ls=':', lw=1.2, zorder=5)
ax.set(xlim=(-.5, 7.5), ylim=(0, 105), yticks=[0, 25, 50, 75, 100],
       xticks=x, xticklabels=mp.index, ylabel='Accuracy (%)', xlabel='Abstract pattern')
ax.tick_params(axis='x', labelsize=10, pad=8)
ax.grid(axis='y', color='#e8edf0', lw=.7)
ax.set_axisbelow(True)
handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
# Legend reads across rows in the same order as the bars.
indices = np.arange(9).reshape(3, 3).T.flatten()
fig.legend(handles=[handles[i] for i in indices], loc='upper left',
           bbox_to_anchor=(.06, .91), ncol=3, frameon=False,
           columnspacing=2.8, handlelength=1.2, fontsize=11)
fig.text(.065, .055, 'Human bars: equally weighted participant means (N = 25); error bars: pointwise 95% bootstrap intervals.',
         fontsize=10, color='#657581')
fig.text(.065, .027, 'Model bars: observed accuracy for each pattern. Dotted line: 25% chance. Model uncertainty is not shown.',
         fontsize=10, color='#657581')
save(fig, 'figure_1b_accuracy_grouped_bars')
print(O)
