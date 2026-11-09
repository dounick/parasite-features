import argparse
import csv
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene import IntervenableConfig, IntervenableModel, RepresentationConfig, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'syntax'))
from interventions.interventions import LowRankInvertibleUnconstrainedIntervention

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:3')
parser.add_argument('--layer', type=int, default=22)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--train-pairs', type=int, default=600)
parser.add_argument('--test-pairs', type=int, default=400)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', required=True)
args = parser.parse_args()
set_seed(args.seed)
rng = random.Random(args.seed)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
vocab = json.loads(Path(__file__).with_name('vocab.json').read_text())

def sentence(noun, adjectives):
    return f'The {noun} was {adjectives[0]} and {adjectives[1]}.'

def prompt(noun, adjectives):
    return sentence(noun, adjectives) + ' Overall, the sentiment is'

def pairs(split, size):
    rows = []
    words = vocab[split]
    for i in range(size):
        base_class = (i // 2) % 2
        across = i % 2
        source_class = base_class ^ across
        base_noun = rng.choice(words['nouns'][base_class])
        source_noun = rng.choice([n for n in words['nouns'][source_class] if n != base_noun])
        adjectives = rng.sample(words['adjectives'][base_class], 2)
        rows.append(dict(base_noun=base_noun, source_noun=source_noun,
                         adjectives=adjectives, base_class=base_class,
                         source_class=source_class, across=bool(across)))
    rng.shuffle(rows)
    return rows

train = pairs('train', args.train_pairs * 3)
test = pairs('test', args.test_pairs * 3)
(out / 'config.json').write_text(json.dumps(vars(args), indent=2))
name = 'meta-llama/Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
label_ids = [tokenizer.encode(word, add_special_tokens=False) for word in [' positive', ' negative']]
if any(len(ids) != 1 for ids in label_ids):
    raise ValueError(f'Expected single-token sentiment labels: {label_ids}')
labels = torch.tensor([ids[0] for ids in label_ids], device=args.device)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, local_files_only=True).to(args.device)
model.eval().requires_grad_(False)
config = IntervenableConfig(model_type=type(model), representations=[RepresentationConfig(
    layer=args.layer, component='block_output', low_rank_dimension=2)],
    intervention_types=LowRankInvertibleUnconstrainedIntervention)
intervenable = IntervenableModel(config, model)
intervenable.set_device(args.device)
intervenable.disable_model_gradients()
intervenable.eval()
params = [p for intervention in intervenable.interventions.values() for p in intervention.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=args.lr)

def encode(nouns, adjective_pairs):
    texts = [prompt(n, a) for n, a in zip(nouns, adjective_pairs)]
    encoded = tokenizer(texts, padding=True, return_tensors='pt').to(args.device)
    positions = []
    for noun, adjectives, text, mask in zip(nouns, adjective_pairs, texts, encoded.attention_mask):
        offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
        period = len(sentence(noun, adjectives)) - 1
        pos = next(i for i, (start, end) in enumerate(offsets) if start <= period < end)
        positions.append([pos + encoded.input_ids.shape[1] - int(mask.sum())])
    return encoded, positions

def forward(rows, original=False):
    adjectives = [r['adjectives'] for r in rows]
    base, bp = encode([r['base_noun'] for r in rows], adjectives)
    source, sp = encode([r['source_noun'] for r in rows], adjectives)
    before, after = intervenable(base, [source], {'sources->base': ([sp], [bp])},
                                 output_original_output=original)
    return before, after

behavior = []
for cls, nouns in enumerate(vocab['test']['nouns']):
    for noun in nouns:
        for sentiment in range(2):
            for _ in range(4):
                behavior.append((noun, rng.sample(vocab['test']['adjectives'][sentiment], 2), sentiment))
correct = 0
with torch.no_grad():
    for start in range(0, len(behavior), args.batch_size):
        batch = behavior[start:start + args.batch_size]
        encoded, _ = encode([b[0] for b in batch], [b[1] for b in batch])
        scores = model(**encoded).logits[:, -1, labels].float()
        correct += (scores.argmax(-1).cpu() == torch.tensor([b[2] for b in batch])).sum().item()
print('Held-out behavioral accuracy (fully crossed noun class / sentiment):', correct / len(behavior), flush=True)

calibration_rows = []
for nouns in vocab['train']['nouns']:
    for noun in nouns:
        for sentiment in range(2):
            for _ in range(4):
                calibration_rows.append((noun, rng.sample(vocab['train']['adjectives'][sentiment], 2), sentiment))
calibration_gaps = []
with torch.no_grad():
    for start in range(0, len(calibration_rows), args.batch_size):
        batch = calibration_rows[start:start + args.batch_size]
        encoded, _ = encode([r[0] for r in batch], [r[1] for r in batch])
        scores = model(**encoded).logits[:, -1, labels].float()
        calibration_gaps.extend((scores[:, 0] - scores[:, 1]).cpu().tolist())
means = [sum(g for r, g in zip(calibration_rows, calibration_gaps) if r[2] == sentiment) /
         sum(r[2] == sentiment for r in calibration_rows) for sentiment in range(2)]
bias = sum(means) / 2
calibration = dict(positive_mean_log_odds=means[0], negative_mean_log_odds=means[1],
                   bias=bias, centered_accuracy=sum((g > bias) == (r[2] == 0) for r, g in zip(calibration_rows, calibration_gaps)) / len(calibration_rows))
(out / 'calibration.json').write_text(json.dumps(dict(summary=calibration, examples=[dict(noun=r[0], adjectives=r[1], sentiment=r[2], log_odds=g) for r, g in zip(calibration_rows, calibration_gaps)]), indent=2))
print('Training-only sentiment calibration:', json.dumps(calibration), flush=True)

def filter_pairs(rows, requested):
    keys = sorted({(r[side + '_noun'], tuple(r['adjectives'])) for r in rows for side in ['base', 'source']})
    scores = {}
    with torch.no_grad():
        for start in range(0, len(keys), args.batch_size):
            batch = keys[start:start + args.batch_size]
            encoded, _ = encode([x[0] for x in batch], [x[1] for x in batch])
            logits = model(**encoded).logits[:, -1, labels].float()
            for key, value in zip(batch, (logits[:, 0] - logits[:, 1]).cpu().tolist()):
                scores[key] = value
    groups = {(cls, across): [] for cls in range(2) for across in [False, True]}
    for row in rows:
        sign = 1 if row['base_class'] == 0 else -1
        margins = {side + '_margin': sign * (scores[(row[side + '_noun'], tuple(row['adjectives']))] - bias) for side in ['base', 'source']}
        if min(margins.values()) >= args.margin:
            groups[(row['base_class'], row['across'])].append(dict(**row, **margins))
    print('Filter candidates by (base_class, across):', {str(key): len(group) for key, group in groups.items()}, flush=True)
    per_group = min(requested // 4, *(len(group) for group in groups.values()))
    if per_group == 0:
        raise ValueError('No balanced dataset remains after baseline filtering')
    retained = [row for group in groups.values() for row in group[:per_group]]
    rng.shuffle(retained)
    stats = dict(candidates=len(rows), pass_by_group={str(key): len(group) for key, group in groups.items()},
                 retained=len(retained), minimum_centered_log_odds=args.margin, bias=bias,
                 nouns=sorted({r[side + '_noun'] for r in retained for side in ['base', 'source']}),
                 adjectives=sorted({word for r in retained for word in r['adjectives']}))
    return retained, stats

train, train_filter = filter_pairs(train, args.train_pairs)
test, test_filter = filter_pairs(test, args.test_pairs)
filter_stats = dict(train=train_filter, test=test_filter)
(out / 'filtering.json').write_text(json.dumps(filter_stats, indent=2))
(out / 'data.json').write_text(json.dumps(dict(train=train, test=test, vocab=vocab), indent=2))
print('Baseline filtering:', json.dumps(filter_stats), flush=True)
for epoch in range(args.epochs):
    rng.shuffle(train)
    total = 0.0
    for start in range(0, len(train), args.batch_size):
        batch = train[start:start + args.batch_size]
        optimizer.zero_grad(set_to_none=True)
        _, after = forward(batch)
        targets = labels[torch.tensor([r['source_class'] for r in batch], device=args.device)]
        loss = F.cross_entropy(after.logits[:, -1].float(), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    print(f'Epoch {epoch + 1}: loss={total / len(train):.6f}', flush=True)
intervenable.save_intervention(str(out / 'alignment'))
records = []
with torch.no_grad():
    for start in range(0, len(test), args.batch_size):
        batch = test[start:start + args.batch_size]
        before, after = forward(batch, original=True)
        before_lp = before.logits[:, -1].float().log_softmax(-1)[:, labels].cpu()
        after_lp = after.logits[:, -1].float().log_softmax(-1)[:, labels].cpu()
        for row, b, a in zip(batch, before_lp.tolist(), after_lp.tolist()):
            effect = (a[0] - a[1]) - (b[0] - b[1])
            records.append(dict(**row, before_positive=b[0], before_negative=b[1],
                                after_positive=a[0], after_negative=a[1],
                                delta_log_odds=effect,
                                target_directed_effect=effect * (1 if row['source_class'] == 0 else -1),
                                target_correct=int((a[0] > a[1]) == (row['source_class'] == 0)),
                                centered_target_correct=int((a[0] - a[1] > bias) == (row['source_class'] == 0))))
with (out / 'test.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
metrics = dict(unfiltered_behavioral_accuracy=correct / len(behavior), filtering=filter_stats, calibration=calibration, label_token_ids=labels.tolist())
for across, prefix in [(True, 'across'), (False, 'within')]:
    group = [r for r in records if r['across'] == across]
    metrics[prefix] = dict(n=len(group), mean_target_directed_effect=sum(r['target_directed_effect'] for r in group) / len(group),
                          mean_absolute_log_odds_effect=sum(abs(r['delta_log_odds']) for r in group) / len(group),
                          raw_target_accuracy=sum(r['target_correct'] for r in group) / len(group),
                          centered_target_accuracy=sum(r['centered_target_correct'] for r in group) / len(group))
(out / 'results.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2), flush=True)
