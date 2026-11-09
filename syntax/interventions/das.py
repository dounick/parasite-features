import torch
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from pyvene import (
    IntervenableModel,
    set_seed,
    count_parameters,
)
import pandas as pd
import numpy as np

from interventions.configs import get_config
from interventions.metrics import compute_metrics, calculate_loss
from utils import get_last_token
import gc
import wandb


def run_das_experiment(
    intervention_name,
    model,
    layer,
    component="block_output",
    trainset=None,
    evalset=None,
    n_train=None,
    intervention_pos=None,
    epochs=1,
    low_rank_dimension=1,
    intervention_dim=1,
    include_same_type=False,
    learning_rate=1e-3,
    temperature_scheduling=False,
    gradient_accumulation_steps=4,
    transformation=None,
    num_layers=None,
    hidden_dim=None,
    config=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_intervention=False,
):
    """
    Run DAS experiment matching the notebook's approach.

    Args:
        intervention_name: 'low_rank', 'boundless', or 'das'
        model: The language model
        tokenizer: The tokenizer
        layer: Layer number for intervention
        component: Component type (default: "block_output")
        trainset: Training data
        evalset: Test data
        intervention_pos: Position to apply intervention (default: 60)
        epochs: Number of training epochs (default: 1)
        low_rank_dimension: Dimension for low rank interventions (default: 1)
        learning_rate: Learning rate (default: 1e-3)
        temperature_scheduling: Whether to use temperature scheduling (default: True)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4),
        transformation: Optional transformation to apply (e.g., 'cubic', 'translation')
    """
    # Set seed for reproducibility
    set_seed(42)

    intervention_layer = layer

    # Optional wandb integration
    use_wandb = config.get("use_wandb", False) if isinstance(config, dict) else False
    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=config.get("wandb_project", "das-experiments"),
            name=config.get("wandb_run_name", None) if isinstance(config, dict) else None,
            id=config.get("wandb_run_id", None) if isinstance(config, dict) else None,
            config={
                "intervention_name": intervention_name,
                "intervention_layer": intervention_layer,
                "component": component,
                "low_rank_dimension": low_rank_dimension,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "temperature_scheduling": temperature_scheduling,
                "task": "price-tagging",
                "transformation": transformation,
                "seed": 42
            }
        )

    # Force clean model state
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create configuration matching the notebook
    config, layer_type = get_config(
        intervention_name=intervention_name,
        model_type=type(model),
        layer=intervention_layer,
        component=component,
        low_rank_dimension=low_rank_dimension,
        intervention_dim=intervention_dim,
        n_layers=num_layers,
        hidden_dim=hidden_dim,
        latent_dim=model.config.hidden_size
    )

    # Create intervenable model
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    # Initialize gradients
    intervenable.set_zero_grad()

    # Setup optimizer — t_total should count actual optimizer steps (batches),
    # not samples, so the warmup schedule is correct.
    t_total = len(trainset) * epochs
    warm_up_steps = 0.1 * t_total
    optimizer_params = []

    for k, v in intervenable.interventions.items():
        layer = getattr(v, layer_type)
        optimizer_params += [{"params": layer.parameters()}]
        # Add boundary parameters for boundless interventions
        if intervention_name == "boundless":
            optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]

    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    # Neural interventions (RealNVP, RevNet) are sensitive to lr scheduling —
    # use constant lr. Linear interventions benefit from warmup.
    from utils import neural_interventions
    if intervention_name in neural_interventions:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
        )

    # Temperature scheduling setup
    total_step = 0
    target_total_step = len(trainset) * epochs

    if temperature_scheduling:
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, target_total_step)
            .to(torch.bfloat16)
            .to(device)
        )
        intervenable.set_temperature(temperature_schedule[total_step])


    intervenable.model.train()  # train enables drop-off but no grads
    # print("llama trainable parameters: ", count_parameters(intervenable.model))
    # print("intervention trainable parameters: ", intervenable.count_parameters())

    total_loss = torch.tensor(0.0).to(intervenable.get_device())
    epoch_iterator = trange(epochs, desc="Epochs", leave=False)
    for epoch in epoch_iterator:
        for batch in tqdm(trainset, desc="Training", leave=False):
            pos_interv = [[x[intervention_pos] for x in y] for y in batch.compute_pos('last')] # HARD CODED AS LAST RIGHT NOW
            pos_interv = [[x] for x in pos_interv]  # Adjust shape to match expected input

            base_outputs, counterfactual_outputs = intervenable(
                batch.base,
                [batch.src],
                {"sources->base": pos_interv},
                output_original_output=True
            )

            eval_metrics = compute_metrics(
                get_last_token(base_outputs.logits.detach().cpu(), attention_mask=batch.base['attention_mask']),
                get_last_token(counterfactual_outputs.logits.detach().cpu(), attention_mask=batch.base['attention_mask']),
                batch.base_labels.cpu(),
                batch.src_labels.cpu()
            )

            # Loss and backprop - matching notebook's calculate_loss
            loss = calculate_loss(get_last_token(counterfactual_outputs.logits, attention_mask=batch.base['attention_mask']), batch.src_labels)
            total_loss += loss
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"], "odds": eval_metrics["odds"]})

            # Log to wandb if available
            if wandb is not None:
                try:
                    run.log({
                        "train/loss": loss.item(),
                        "train/accuracy": eval_metrics["accuracy"],
                        "train/odds": eval_metrics["odds"],
                        "train/epoch": epoch,
                        "train/step": total_step
                    }, step=total_step)
                except Exception:
                    # Don't fail training for wandb issues
                    pass

            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    total_loss.backward()
                    total_loss = torch.tensor(0.0).to(intervenable.get_device())
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    if temperature_scheduling:
                        intervenable.set_temperature(temperature_schedule[total_step])

            # Clean up memory after each training step
            del counterfactual_outputs, loss

            total_step += 1

    torch.cuda.empty_cache()
    gc.collect()
    intervenable.model.eval()  # eval disables drop-off

    # Evaluation loop matching the notebook
    eval_iterator = tqdm(evalset, desc="Evaluating", position=2, leave=False)
    final_metrics = []
    with torch.no_grad():
        for batch in eval_iterator:
            # Compute the intervention positions for this batch (matching training)
            pos_interv = [[x[intervention_pos] for x in y] for y in batch.compute_pos('last')]  # HARD CODED AS LAST RIGHT NOW
            pos_interv = [[x] for x in pos_interv]  # Adjust shape to match expected input

            # Pass the per-batch positions (pos_interv) rather than the scalar intervention_pos
            base_outputs, counterfactual_outputs = intervenable(
                batch.base,
                [batch.src],
                {"sources->base": pos_interv},  # swap pos-th token using computed positions
                output_original_output=True,
            )

            # Compute metrics per batch and append
            eval_metrics = compute_metrics(
                get_last_token(base_outputs.logits.detach().cpu(), attention_mask=batch.base['attention_mask']),
                get_last_token(counterfactual_outputs.logits.detach().cpu(), attention_mask=batch.base['attention_mask']),
                batch.base_labels.cpu(),
                batch.src_labels.cpu(),
            )
            final_metrics.append(eval_metrics)
            eval_iterator.set_postfix({"eval_accuracy": eval_metrics.get("accuracy"), "eval_odds": eval_metrics.get("odds")})
            # Print a running summary for visibility (will show last batch when finished)


    final_accuracy = np.mean([m['accuracy'] for m in final_metrics])
    final_odds = np.mean([m['odds'] for m in final_metrics])
    # Final wandb logging
    if wandb is not None:
        try:
            run.log({
                "eval/accuracy": final_accuracy,
                "eval/odds": final_odds,
            })
            # finish the run if we initialized it here
            if run is not None:
                run.finish()
        except Exception:
            pass

    if save_intervention:
        intervenable.save_intervention(f"interventions/cached_interventions/{intervention_name}_layer{intervention_layer}_pos{intervention_pos}_dim{low_rank_dimension}.pt")
    return {"accuracy": final_accuracy, "odds": final_odds, "layer": intervention_layer, "position": intervention_pos,
            "intervention_name": intervention_name, "low_rank_dimension": low_rank_dimension, "intervention_dim": intervention_dim,
            "include_same_type": include_same_type, "component": component}


# Legacy function for backward compatibility
def run_das_experiment_legacy(intervention, model, tokenizer, target_representation, train_dataloader, test_dataloader, intervention_pos, epochs=3, max_intervention_dimension=40):
    """
    Legacy wrapper for backward compatibility.
    """
    return run_das_experiment(
        intervention_name=intervention,
        model=model,
        tokenizer=tokenizer,
        layer=target_representation.layer,
        component=target_representation.component,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        intervention_pos=intervention_pos,
        epochs=epochs,
        low_rank_dimension=1
    )["accuracy"]
