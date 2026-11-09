import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
root = Path(__file__).resolve().parents[1]
parser.add_argument('--results', type=Path, default=root / 'reference_results.csv')
parser.add_argument('--output', type=Path, default=root / 'runs/carry_shift_accuracy')
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.results.open() as f:
    scores = {r['experiment']: float(r['value']) for r in csv.DictReader(f)
              if r['layer'] == '20' and r['method'] == 'rotation + linear' and r['metric'] == 'IIA'}
rows = [{'n': n, 'accuracy': scores[f'carry shift ±{n}' if n < 10 else 'carry-one']}
        for n in range(1, 11)]

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
fig, ax = plt.subplots(figsize=(4.5, 2.65))
ax.bar([r['n'] for r in rows[:9]], [r['accuracy'] for r in rows[:9]],
       width=0.68, color='#0066CC')
ax.bar([10], [rows[9]['accuracy']], width=0.68, color='#55A82A')
ax.set_xticks(range(1, 11))
ax.set_xlim(0.35, 10.65)
ax.set_ylim(0.90, 1.0)
ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.0])
ax.set_yticklabels(['90%', '92%', '94%', '96%', '98%', '100%'])
ax.set_xlabel(r'$n$')
ax.set_ylabel('Accuracy')
ax.set_title('rotation+linear at L20', pad=9)
ax.tick_params(length=3)
fig.tight_layout(pad=0.6)
for extension in ['pdf', 'png']:
    fig.savefig(args.output.with_suffix('.' + extension), dpi=300, bbox_inches='tight')
plt.close(fig)
