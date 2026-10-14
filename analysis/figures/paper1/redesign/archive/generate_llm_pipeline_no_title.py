from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = '/tmp/elife-mpl-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
B = Path(__file__).resolve().parent
O = B / 'llm_pipeline'
O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 11})
ink = '#20303b'
muted = '#607480'
blue = '#0072B2'
green = '#008778'
gold = '#bd7100'
f = plt.figure(figsize=(15, 10), facecolor='white')
bg = f.add_axes([0, 0, 1, 1])
bg.set(xlim=(0, 1), ylim=(0, 1))
bg.axis('off')

def text(x, y, s, size=11, color=ink, weight='normal', ha='left'):
    bg.text(x, y, s, fontsize=size, color=color, fontweight=weight, va='top', ha=ha, linespacing=1.4)

def arrow(x, y, u, v):
    bg.add_patch(FancyArrowPatch((x, y), (u, v), arrowstyle='-|>', mutation_scale=14, color=muted, lw=1.5))

def card(x, y, num, title):
    bg.add_patch(FancyBboxPatch((x, y), 0.29, 0.35, boxstyle='round,pad=.008,rounding_size=.012', facecolor='#f6f8fa', edgecolor='#dce4e9'))
    text(x + 0.012, y + 0.327, num, 15, blue, 'bold')
    text(x + 0.043, y + 0.325, title, 12, ink, 'bold')

def matrix(rect, arr, cmap='Blues'):
    a = f.add_axes(rect)
    a.imshow(arr, cmap=cmap, aspect='auto')
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    return a
rng = np.random.default_rng(13)
for x, n, t in zip([0.04, 0.355, 0.67], ['1', '2', '3'], ['Present one task trial', 'Read internal activations', 'Build trial representations']):
    card(x, 0.505, n, t)
for x, n, t in zip([0.04, 0.355, 0.67], ['4', '5', '6'], ['Select a task-informative layer', 'Construct the pattern RDM', 'Compare with human geometry']):
    card(x, 0.105, n, t)
text(0.055, 0.792, 'Instructions + one solved example', 10, muted)
text(0.055, 0.752, 'Target sequence', 10, blue, 'bold')
bg.add_patch(FancyBboxPatch((0.053, 0.677), 0.263, 0.047, boxstyle='round,pad=.003', facecolor='#dceef8', edgecolor='none'))
text(0.06, 0.711, 'star star star guitar star star star ?', 9, ink)
text(0.055, 0.657, 'Options: truck / star / cube / guitar', 9)
text(0.055, 0.615, 'Tokenize the full prompt', 11, ink, 'bold')
text(0.055, 0.577, '400 independent trials per model\nNo previous-trial context', 10, muted)
for k in range(4):
    y = 0.752 - k * 0.039
    bg.add_patch(FancyBboxPatch((0.375, y), 0.098, 0.029, boxstyle='round,pad=.002', facecolor='#dceef8', edgecolor=blue, lw=0.6))
    text(0.424, y + 0.023, 'Block ' + str(k + 1) if k < 3 else '… Block L', 9, ink, ha='center')
    arrow(0.479, y + 0.014, 0.502, y + 0.014)
    matrix([0.507, y, 0.117, 0.028], rng.normal(size=(3, 15)))
text(0.375, 0.8, 'Transformer', 10, ink, 'bold')
text(0.507, 0.8, 'Layer outputs', 10, blue, 'bold')
text(0.375, 0.578, 'Retain the target-sequence token span\nHidden features at every layer', 10, muted)
matrix([0.689, 0.674, 0.101, 0.098], rng.normal(size=(8, 18)))
text(0.687, 0.792, 'Tokens × hidden features', 10, blue, 'bold')
arrow(0.798, 0.721, 0.83, 0.721)
matrix([0.835, 0.703, 0.1, 0.035], rng.normal(size=(1, 144)))
text(0.834, 0.682, 'Flatten', 10, ink, 'bold')
text(0.689, 0.628, 'One vector per trial, per layer', 11, ink, 'bold')
text(0.689, 0.586, 'Keep token-position information\nStack trial vectors for each layer', 10, muted)
labels = np.repeat(np.arange(8), 8)
ref = (labels[:, None] != labels[None, :]).astype(float)
matrix([0.059, 0.263, 0.075, 0.103], ref)
text(0.055, 0.397, 'Task-design RDM', 10, ink, 'bold')
text(0.055, 0.245, 'Same pattern: 0\nDifferent pattern: 1', 8, muted)
a = f.add_axes([0.172, 0.274, 0.136, 0.093])
xx = np.linspace(0, 1, 25)
yy = 0.7 * np.sin(np.pi * xx) ** 1.3
a.plot(xx, yy, color=blue, lw=2)
a.scatter([xx[12]], [yy[12]], color=gold, s=40, zorder=3)
a.set(xticks=[0, 1], yticks=[], xlabel='Relative layer depth')
a.xaxis.label.set_size(8)
a.tick_params(labelsize=8)
a.spines[['top', 'right', 'left']].set_visible(False)
text(0.169, 0.397, 'RSA at each layer', 10, ink, 'bold')
text(0.055, 0.19, 'Correlate 400 × 400 trial RDMs with the\ntask-design RDM; retain the best layer.', 10)
text(0.055, 0.132, 'Layer selection uses no human EEG.', 10, green, 'bold')
features = rng.normal(size=(8, 24))
rdm = 1 - np.corrcoef(features)
matrix([0.375, 0.264, 0.097, 0.104], features)
text(0.374, 0.397, '8 pattern-mean vectors', 10, ink, 'bold')
arrow(0.48, 0.315, 0.51, 0.315)
matrix([0.524, 0.264, 0.083, 0.104], rdm)
text(0.523, 0.247, '8 × 8 RDM', 9, blue, 'bold')
text(0.375, 0.206, 'Average trial vectors within each pattern\nat the selected layer.', 10)
text(0.375, 0.149, 'Dissimilarity = 1 − Pearson r\nRepeat separately for each model.', 10, muted)
for x, title, mat in [(0.689, 'LLM', rdm), (0.825, 'Human EEG', 1 - np.corrcoef(features + 0.5 * rng.normal(size=features.shape)))]:
    text(x, 0.397, title, 10, ink, 'bold')
    ax = matrix([x, 0.275, 0.074, 0.1], mat)
    ax.add_patch(Polygon([(0.5, -0.5), (7.5, -0.5), (7.5, 6.5)], fill=False, edgecolor=gold, lw=1.5))
arrow(0.768, 0.324, 0.82, 0.324)
text(0.689, 0.247, 'Extract the same 28 unique pattern pairs', 10)
text(0.689, 0.201, 'Pearson correlation between RDM entries', 10, blue, 'bold')
text(0.689, 0.157, 'One RSA value per model, EEG measure\nand regional group.', 10, muted)
arrow(0.332, 0.686, 0.35, 0.686)
arrow(0.647, 0.686, 0.665, 0.686)
arrow(0.332, 0.286, 0.35, 0.286)
arrow(0.647, 0.286, 0.665, 0.286)
text(0.5, 0.484, 'Repeat across layers → choose a layer → compare pattern geometry', 10, muted, ha='center')
text(0.04, 0.06, 'Answers are scored separately for behavioral accuracy; the RSA features are internal activations.', 11, ink, 'bold')
text(0.04, 0.027, 'Schematic illustration: token grids, matrices and layer trajectory are not measured data. Eight models are analyzed separately.', 9, muted)
out = O / 'llm_representation_extraction_pipeline_no_title'
f.savefig(out.with_suffix('.pdf'), bbox_inches=matplotlib.transforms.Bbox.from_extents(0.3, 0.1, 14.7, 8.8))
f.savefig(out.with_suffix('.png'), dpi=180, bbox_inches=matplotlib.transforms.Bbox.from_extents(0.3, 0.1, 14.7, 8.8))
plt.close(f)
print(out.with_suffix('.pdf'))
