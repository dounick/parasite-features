# Parasite Features

Repo for the paper `Parasite Features: Causal Abstraction Through Spurious Pathways`, to appear in NeuRIPS 2026.

Python 3.10 was used. Install `requirements.txt` in a CUDA-enabled environment.

## Syntax

```bash
cd syntax
mkdir -p interventions/cached_interventions
python experiment.py --model EleutherAI/pythia-1b --template agr_sv_num_pp --intervention_types low_rank low_rank_invertible_unconstrained --layer 2 --position 1 --include_same_type --epochs 5 --total_samples 400 --batch_size 4 --learning_rate 0.005 --seed 42 --save_intervention --wandb
```

This is the baseline/faithful-agreement run. For parasitic agreement use `--template agr_sv_num_pp_gender_parasite`.

## Carry-one

```bash
cd carry
export PYTHONPATH=.:CausalAbstraction
python experiment_run_RotationLinear.py Carry-One 20 --layer 18 --device cuda:0 --output ../runs/carry18
python experiment_run_RotationLinear.py Carry-One 20 --layer 20 --device cuda:0 --output ../runs/carry20
```

For +- n runs, add `--shift n` (1–9).

## Sentiment

```bash
python sentiment/experiment.py --output runs/sentiment-l22-half-vocab
```

The default layer-22 run uses 600 training / 400 test pairs with disjoint noun and adjective vocabularies.
