import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
root = Path(__file__).resolve().parents[1]
parser.add_argument('--results', type=Path, default=root / 'reference_results.csv')
parser.add_argument('--output', type=Path, default=root / 'runs/sentiment_odds')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.results.open() as f:
    scores = {(r['experiment'], r['metric']): float(r['value']) for r in csv.DictReader(f)
              if r['layer'] == '22' and r['method'] == 'rotation + linear'}
values = [scores['sentiment across', 'target-directed log-odds change'],
          scores['sentiment within', 'mean absolute log-odds change']]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(3.45, 2.45))
bars = ax.bar([0, 1], values, width=0.52, color='#0066CC')
ax.set_xticks([0, 1], ['Across class\n(signed)', 'Within class\n(absolute)'])
ax.set_ylabel('Odds')
ax.set_title('rotation+linear at L22', pad=9)
ax.set_ylim(0, 4.25)
ax.set_yticks([0, 1, 2, 3, 4])
ax.tick_params(length=3)
ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=10)
fig.tight_layout(pad=0.6)
for extension in ['pdf', 'png']:
    fig.savefig(args.output.with_suffix('.' + extension), dpi=300, bbox_inches='tight')
plt.close(fig)
