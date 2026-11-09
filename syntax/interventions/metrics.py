import torch
from torch.nn import CrossEntropyLoss


def calculate_loss(logits: torch.tensor, label: torch.tensor) -> torch.tensor:
    """Calculate cross entropy between logits and a single target label (can be batched)"""
    shift_logits = logits.contiguous()
    shift_labels = label.to(shift_logits.device)
    loss = CrossEntropyLoss()(shift_logits, shift_labels)
    return loss


def compute_metrics(base_logits, counterfactual_logits, base_labels, counterfactual_labels):
    """
    Compute metrics matching the notebook's approach.
    Compares the last token predictions with the true labels.
    """
    base_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    counterfactual_probs = torch.nn.functional.log_softmax(counterfactual_logits, dim=-1)

    base_base_probs = torch.gather(base_probs, 1, base_labels.unsqueeze(-1)).squeeze(-1)
    base_src_probs = torch.gather(base_probs, 1, counterfactual_labels.unsqueeze(-1)).squeeze(-1)
    src_base_probs = torch.gather(counterfactual_probs, 1, base_labels.unsqueeze(-1)).squeeze(-1)
    src_src_probs = torch.gather(counterfactual_probs, 1, counterfactual_labels.unsqueeze(-1)).squeeze(-1)

    correct_tensor = (src_src_probs > base_src_probs) & (base_base_probs > src_base_probs)
    correct_count = int(correct_tensor.sum().item())

    total_count = base_logits.size(0)
    odds_tensor = base_base_probs - base_src_probs + src_src_probs - src_base_probs
    odds_sum = float(odds_tensor.sum().item())

    accuracy = round(correct_count / total_count, 2)
    odds = round(odds_sum / total_count, 2)

    return {"accuracy": accuracy, "odds": odds}
