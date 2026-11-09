import os, pandas as pd
import torch
import torch.nn.functional as F
lsm = F.log_softmax
sm = F.softmax


DEFAULT_CONFIG = {
    "epochs": 5,
    "batch_size": 4,
    "total_samples": 400,
    "low_rank_dimension": 1,
    "intervention_dim": 1,
    "include_same_type": False,
    "learning_rate": 5e-3,
    "gradient_accumulation_steps": 1,
    "cache_dir": "../../../shared/hf_cache",
    "baseline_accuracy": 0.9859,
}

neural_interventions = ['realnvp', 'revnet']


def embed_to_distrib(model, embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    if "gpt2" in model.config.architectures[0].lower():
        vocab = torch.matmul(embed, model.wte.weight.t())
        if logits:
            return vocab
        return lsm(vocab) if log else sm(vocab)


def get_last_token(logits, attention_mask):
    # Ensure attention_mask and index tensors are on the same device as logits
    device = logits.device
    am = attention_mask.to(device)
    # compute last token index (clamp to 0 to avoid -1 if mask is empty)
    last_token_indices = (am.sum(1) - 1).clamp(min=0).long()
    batch_indices = torch.arange(logits.size(0), device=device).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)
