"""
Unified intervention visualization script.

Instead of selecting fixed base prompts and sweeping grids for each,
this script:
1. Creates a grid of arbitrary points in the intervention space
2. For each grid point, randomly samples a base example
3. Performs the intervention from the base to that grid point
4. Records the result

This produces a single (noisier) landscape showing aggregated behavior.
"""
import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from interventions.configs import get_config, collect_config
from pyvene import IntervenableModel
from experiment import get_data
from utils import get_last_token
try:
    import plotnine as p9
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: plotnine not available, figures will not be generated")


def load_models_and_intervention(model_name, intervention_name, intervention_layer, position,
                                  low_rank_dimension, num_layers, hidden_dim, device, intervention_folder='regular'):
    """Load model, tokenizer, and intervention"""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(model_name)
    gpt.to(device)
    gpt.eval()

    folder_display = "root" if intervention_folder == "root" else f"{intervention_folder}/"
    print(f"Loading intervention: {intervention_name} at layer {intervention_layer}, position {position} from {folder_display}...")
    config, layer_type = get_config(
        intervention_name=intervention_name,
        model_type=type(gpt),
        layer=intervention_layer,
        component='block_output',
        low_rank_dimension=low_rank_dimension,
        n_layers=num_layers,
        hidden_dim=hidden_dim,
        latent_dim=gpt.config.hidden_size
    )

    intervenable = IntervenableModel(config, gpt)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    intervenable.set_zero_grad()

    # Determine save_dim based on intervention type
    # low_rank uses dim2 (2D projection), others use dim1
    save_dim = 2 if intervention_name == 'low_rank' else 1

    # Handle root folder vs subfolders
    if intervention_folder == 'root':
        intervention_path = f"interventions/cached_interventions/{intervention_name}_layer{intervention_layer}_pos{position}_dim{save_dim}.pt"
    else:
        intervention_path = f"interventions/cached_interventions/{intervention_folder}/{intervention_name}_layer{intervention_layer}_pos{position}_dim{save_dim}.pt"

    intervenable.load_intervention(intervention_path)

    # Extract intervention layers
    intervention_keys = list(intervenable.interventions.keys())
    intervention = intervenable.interventions[intervention_keys[0]]

    if intervention_name == 'low_rank_invertible_unconstrained':
        other_layer = intervention.unconstrained_layer
        interchange_dim = 1
    elif intervention_name == 'low_rank_invertible_unconstrained_2d':
        other_layer = intervention.unconstrained_layer
        interchange_dim = 2
    elif intervention_name == 'low_rank_invertible_rotated':
        other_layer = intervention.rotate_layer_two
        interchange_dim = 1
    elif intervention_name == 'low_rank_invertible_realnvp':
        other_layer = intervention.realNVPlayer
        interchange_dim = 1
    elif intervention_name == 'low_rank_invertible_revnet':
        other_layer = intervention.revNetLayer
        interchange_dim = 1
    elif intervention_name == 'low_rank':
        # low_rank is just a 2D projection without unconstrained layer
        # Use identity transformation in 2D space
        other_layer = None
        interchange_dim = 2
    else:
        other_layer = None
        interchange_dim = 1

    rotate_layer = intervention.rotate_layer

    # Collection model
    collection_config = collect_config(
        model_type=type(gpt),
        layer=intervention_layer,
        component='block_output'
    )
    collect_gpt = IntervenableModel(collection_config, gpt)
    collect_gpt.set_device(device)
    collect_gpt.disable_model_gradients()

    return gpt, tokenizer, intervenable, collect_gpt, rotate_layer, other_layer, interchange_dim


def collect_all_examples(collect_model, dataset, rotate_layer, other_layer, position, tokenizer, which_space, device):
    """
    Collect all examples from the dataset with their activations and transformed coordinates.
    Returns a list of dictionaries, each containing the example data needed for interventions.
    """
    examples = []

    for batch in tqdm(dataset, desc="Collecting all examples"):
        pos_interv = [[x[position] for x in y] for y in batch.compute_pos('last')]
        pos_interv = [[x] for x in pos_interv]

        collected = collect_model(
            base=batch.base,
            sources=None,
            unit_locations={"sources->base": pos_interv}
        )[0][-1][0].squeeze(1)

        with torch.no_grad():
            rotated = rotate_layer(collected)
            if other_layer is not None:
                transformed = other_layer(rotated)
            else:
                transformed = rotated

        for idx in range(len(batch.base_labels)):
            base_label = batch.base_labels[idx].item()
            source_label = batch.src_labels[idx].item()

            # Get token position for this example
            token_pos = pos_interv[0][0]
            if isinstance(token_pos, list):
                token_pos = token_pos[idx] if idx < len(token_pos) else token_pos[0]

            # Get the base input for this example
            base_input = {k: v[idx:idx+1] for k, v in batch.base.items()}

            if which_space == 'projection':
                base_x = rotated[idx, 0].item()
                base_y = rotated[idx, 1].item()
            else:
                base_x = transformed[idx, 0].item()
                base_y = transformed[idx, 1].item()

            examples.append({
                'base_input': base_input,
                'base_activation': collected[idx:idx+1],
                'base_label': base_label,
                'source_label': source_label,
                'base_label_str': tokenizer.decode([base_label]),
                'source_label_str': tokenizer.decode([source_label]),
                'token_position': token_pos,
                'base_x': base_x,
                'base_y': base_y,
                'rotated': rotated[idx:idx+1],
                'transformed': transformed[idx:idx+1] if other_layer is not None else rotated[idx:idx+1]
            })

    return examples


def intervene_single_point(gpt_model, base_input, base_activation, base_label, source_label,
                           rotate_layer, other_layer, intervention_layer, token_position,
                           grid_x, grid_y, base_transformed, only_x, which_space, device, tokenizer):
    """
    Perform intervention at a single grid point using the given base example.
    """
    # Define target verb tokens
    verb_tokens = {
        tokenizer.encode(' is', add_special_tokens=False)[0]: ' is',
        tokenizer.encode(' are', add_special_tokens=False)[0]: ' are',
        tokenizer.encode(' has', add_special_tokens=False)[0]: ' has',
        tokenizer.encode(' have', add_special_tokens=False)[0]: ' have',
        tokenizer.encode(' was', add_special_tokens=False)[0]: ' was',
        tokenizer.encode(' were', add_special_tokens=False)[0]: ' were',
    }
    verb_token_ids = torch.tensor(list(verb_tokens.keys()), device=device)

    # Get base coordinates
    base_x = base_transformed[0, 0].item()
    base_y = base_transformed[0, 1].item()

    # Define singular and plural verb sets
    singular_verbs = [' is', ' has', ' was']
    plural_verbs = [' are', ' have', ' were']

    # Get token IDs for singular and plural verbs
    singular_ids = [k for k, v in verb_tokens.items() if v in singular_verbs]
    plural_ids = [k for k, v in verb_tokens.items() if v in plural_verbs]
    singular_ids_tensor = torch.tensor(singular_ids, device=device)
    plural_ids_tensor = torch.tensor(plural_ids, device=device)

    # BASELINE: Run model WITHOUT intervention
    with torch.no_grad():
        orig_outputs = gpt_model(**base_input)
        orig_logits = orig_outputs.logits[:, -1, :]
        orig_probs = torch.nn.functional.log_softmax(orig_logits, dim=-1)

        base_base_prob = orig_probs[0, base_label].item()
        base_source_prob = orig_probs[0, source_label].item()

        # Compute normalized probabilities over the 6 verbs (before intervention)
        before_verb_logits = orig_logits[0, verb_token_ids]
        before_verb_probs = torch.nn.functional.softmax(before_verb_logits, dim=-1)
        # Map back to get P(singular) and P(plural)
        before_singular_prob = sum(before_verb_probs[i].item() for i, tid in enumerate(verb_token_ids)
                                    if verb_tokens[tid.item()] in singular_verbs)
        before_plural_prob = sum(before_verb_probs[i].item() for i, tid in enumerate(verb_token_ids)
                                  if verb_tokens[tid.item()] in plural_verbs)

        # Create the intervened point
        if only_x:
            intervened_point = torch.tensor([[grid_x, base_y]], device=device, dtype=base_transformed.dtype)
        else:
            intervened_point = torch.tensor([[grid_x, grid_y]], device=device, dtype=base_transformed.dtype)

        # Invert back through the layers
        if which_space == 'other' and other_layer is not None:
            if hasattr(other_layer, 'inverse'):
                intervened_rotated = other_layer.inverse(intervened_point)
            else:
                intervened_rotated = other_layer.realnvp.inverse(intervened_point) if hasattr(other_layer, 'realnvp') else other_layer.revnet.inverse(intervened_point)
            base_rotated = other_layer.inverse(base_transformed) if hasattr(other_layer, 'inverse') else (other_layer.realnvp.inverse(base_transformed) if hasattr(other_layer, 'realnvp') else other_layer.revnet.inverse(base_transformed))
        else:
            intervened_rotated = intervened_point
            base_rotated = base_transformed

        # Compute delta in rotated space
        rotated_delta = intervened_rotated - base_rotated

        # Map back to hidden space
        intervention_delta = torch.matmul(rotated_delta, rotate_layer.weight.T)

        # Apply to base activation
        intervened_activation = base_activation + intervention_delta

        # Hook to inject intervention
        def make_hook(new_activation, pos_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, pos_idx, :] = new_activation
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook_fn

        # Run model with intervention (support different architectures)
        if hasattr(gpt_model, 'gpt_neox'):
            layer_module = gpt_model.gpt_neox.layers[intervention_layer]
        elif hasattr(gpt_model, 'transformer'):
            layer_module = gpt_model.transformer.h[intervention_layer]
        elif hasattr(gpt_model, 'model') and hasattr(gpt_model.model, 'layers'):
            layer_module = gpt_model.model.layers[intervention_layer]
        else:
            raise ValueError(f"Unsupported model architecture: {type(gpt_model)}")
        hook_handle = layer_module.register_forward_hook(make_hook(intervened_activation.squeeze(0), token_position))

        try:
            outputs = gpt_model(**base_input)
        finally:
            hook_handle.remove()

        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.log_softmax(logits, dim=-1)

        top_token = logits.argmax(dim=-1).item()
        src_base_prob = probs[0, base_label].item()
        src_source_prob = probs[0, source_label].item()

        # Compute normalized probabilities over the 6 verbs (after intervention)
        after_verb_logits = logits[0, verb_token_ids]
        after_verb_probs = torch.nn.functional.softmax(after_verb_logits, dim=-1)
        after_singular_prob = sum(after_verb_probs[i].item() for i, tid in enumerate(verb_token_ids)
                                   if verb_tokens[tid.item()] in singular_verbs)
        after_plural_prob = sum(after_verb_probs[i].item() for i, tid in enumerate(verb_token_ids)
                                 if verb_tokens[tid.item()] in plural_verbs)

        # Delta score: positive = shift towards plural, negative = shift towards singular
        # delta = P_before(singular) - P_after(singular) + P_after(plural) - P_before(plural)
        delta_score = (before_singular_prob - after_singular_prob) + (after_plural_prob - before_plural_prob)

        # Most likely verb (using log probs for argmax, but we already have softmax probs)
        most_likely_verb_idx = after_verb_probs.argmax().item()
        most_likely_verb_token_id = verb_token_ids[most_likely_verb_idx].item()
        most_likely_verb = verb_tokens[most_likely_verb_token_id]

        # Feature types
        base_label_str = tokenizer.decode([base_label])
        source_label_str = tokenizer.decode([source_label])

        base_feature = 'singular' if base_label_str in singular_verbs else 'plural'
        source_feature = 'singular' if source_label_str in singular_verbs else 'plural'
        is_flip = (base_feature != source_feature)
        predicted_feature = 'singular' if most_likely_verb in singular_verbs else 'plural'

        # Correctness
        if is_flip:
            correct_feature = (predicted_feature == source_feature)
        else:
            correct_feature = (predicted_feature == base_feature)

        correct_token = (src_source_prob > base_source_prob) and (base_base_prob > src_base_prob)

        return {
            'top_token': top_token,
            'top_token_str': tokenizer.decode([top_token]),
            'most_likely_verb': most_likely_verb,
            'predicted_feature': predicted_feature,
            'is_flip': is_flip,
            'base_label': base_label,
            'base_label_str': base_label_str,
            'base_feature': base_feature,
            'source_label': source_label,
            'source_label_str': source_label_str,
            'source_feature': source_feature,
            'base_base_prob': base_base_prob,
            'base_source_prob': base_source_prob,
            'src_base_prob': src_base_prob,
            'src_source_prob': src_source_prob,
            'correct_token': correct_token,
            'correct_feature': correct_feature,
            'correct': correct_feature,
            'delta_score': delta_score,
            'before_singular_prob': before_singular_prob,
            'before_plural_prob': before_plural_prob,
            'after_singular_prob': after_singular_prob,
            'after_plural_prob': after_plural_prob
        }


def unified_grid_sweep(gpt_model, all_examples, rotate_layer, other_layer, intervention_layer,
                       x_range, y_range, grid_resolution, only_x, which_space, device, tokenizer,
                       seed=42):
    """
    Perform a grid sweep where each grid point uses a randomly selected base example.
    """
    np.random.seed(seed)

    x_vals = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_vals = np.linspace(y_range[0], y_range[1], grid_resolution)

    results = []
    total_points = len(x_vals) * len(y_vals)

    with tqdm(total=total_points, desc="Unified grid sweep") as pbar:
        for x in x_vals:
            for y in y_vals:
                # Randomly select a base example
                example = all_examples[np.random.randint(len(all_examples))]

                result = intervene_single_point(
                    gpt_model=gpt_model,
                    base_input=example['base_input'],
                    base_activation=example['base_activation'],
                    base_label=example['base_label'],
                    source_label=example['source_label'],
                    rotate_layer=rotate_layer,
                    other_layer=other_layer,
                    intervention_layer=intervention_layer,
                    token_position=example['token_position'],
                    grid_x=x,
                    grid_y=y,
                    base_transformed=example['transformed'],
                    only_x=only_x,
                    which_space=which_space,
                    device=device,
                    tokenizer=tokenizer
                )

                result['grid_x'] = x
                result['grid_y'] = y
                result['base_x'] = example['base_x']
                result['base_y'] = example['base_y']
                results.append(result)
                pbar.update(1)

    return pd.DataFrame(results)


def plot_unified_landscape(df_grid, df_datapoints, output_dir, which_space, only_x,
                           same_type_suffix="", constrained_prediction=False,
                           use_token_metric=False, plot_by_class=False):
    """Create unified intervention landscape heatmaps"""
    if not PLOTTING_AVAILABLE:
        return

    color_map = {
        ' is': '#0066CC',
        ' has': '#4D94FF',
        ' was': '#99CCFF',
        ' are': '#CC0000',
        ' have': '#FF4D4D',
        ' were': '#FF9999'
    }

    class_color_map = {
        'singular': '#0066CC',
        'plural': '#CC0000'
    }

    # Choose columns
    if plot_by_class:
        pred_col = 'predicted_feature'
    else:
        pred_col = 'most_likely_verb' if constrained_prediction else 'top_token_str'
    correct_col = 'correct_token' if use_token_metric else 'correct_feature'

    # Plot 1: Correctness heatmap
    df_grid_plot = df_grid.copy()
    df_grid_plot['correct_str'] = df_grid_plot[correct_col].astype(str).map({'True': 'Correct', 'False': 'Incorrect'})
    df_grid_plot['correct_str'] = pd.Categorical(df_grid_plot['correct_str'], categories=['Correct', 'Incorrect'])

    p_correct = (
        p9.ggplot(df_grid_plot) +
        p9.aes(x='grid_x', y='grid_y', fill='correct_str') +
        p9.geom_tile(alpha=1.0) +
        p9.geom_point(
            data=df_datapoints,
            mapping=p9.aes(x='x', y='y', color='label'),
            size=1.5,
            alpha=0.5,
            inherit_aes=False
        ) +
        p9.scale_fill_manual(values={'Correct': '#2ECC40', 'Incorrect': '#FF4136'}) +
        p9.scale_color_manual(values=color_map) +
        p9.labs(
            x='Dimension 1',
            y='Dimension 2',
            title=f'Unified Intervention Correctness ({"Token" if use_token_metric else "Feature"} Level)\n({which_space} space, only_x={only_x}, random base per point)',
            fill='Result',
            color='Datapoint Label'
        ) +
        p9.theme_minimal() +
        p9.theme(
            panel_background=p9.element_rect(fill='white'),
            panel_border=p9.element_rect(color='black'),
            figure_size=(10, 8)
        ) +
        p9.scale_x_continuous(expand=(0, 0)) +
        p9.scale_y_continuous(expand=(0, 0))
    )

    output_path = f"{output_dir}/unified_correctness_{which_space}_onlyx{only_x}{same_type_suffix}.png"
    p_correct.save(output_path, dpi=300, verbose=False)
    print(f"  Saved unified correctness landscape: {output_path}")

    # Plot 2: Predicted token/class heatmap
    all_tokens = df_grid[pred_col].unique()

    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb2hex
    import matplotlib

    if plot_by_class:
        full_color_map = class_color_map.copy()
        pred_title = 'Verb Class'
    else:
        unmapped_tokens = [t for t in all_tokens if t not in color_map]
        if len(unmapped_tokens) > 0:
            try:
                cmap = matplotlib.colormaps.get_cmap('Set2')
            except AttributeError:
                cmap = plt.cm.get_cmap('Set2')
            unmapped_colors = {token: rgb2hex(cmap(i / len(unmapped_tokens))) for i, token in enumerate(unmapped_tokens)}
        else:
            unmapped_colors = {}
        full_color_map = {**color_map, **unmapped_colors}
        pred_title = 'Most Likely Verb' if constrained_prediction else 'Predicted Token'

    p_pred = (
        p9.ggplot(df_grid) +
        p9.aes(x='grid_x', y='grid_y', fill=pred_col) +
        p9.geom_tile(alpha=0.8) +
        p9.geom_point(
            data=df_datapoints,
            mapping=p9.aes(x='x', y='y', color='label'),
            size=1.5,
            alpha=0.5,
            inherit_aes=False
        ) +
        p9.scale_fill_manual(values=full_color_map) +
        p9.scale_color_manual(values=color_map) +
        p9.labs(
            x='Dimension 1',
            y='Dimension 2',
            title=f'Unified {pred_title} Landscape\n({which_space} space, only_x={only_x}, random base per point)',
            fill=pred_title,
            color='Datapoint Label'
        ) +
        p9.theme_minimal() +
        p9.theme(
            panel_background=p9.element_rect(fill='white'),
            panel_border=p9.element_rect(color='black'),
            figure_size=(10, 8)
        ) +
        p9.scale_x_continuous(expand=(0, 0)) +
        p9.scale_y_continuous(expand=(0, 0))
    )

    if plot_by_class:
        constrained_suffix = "_class"
    elif constrained_prediction:
        constrained_suffix = "_verbs"
    else:
        constrained_suffix = ""
    output_path = f"{output_dir}/unified_predictions_{which_space}_onlyx{only_x}{same_type_suffix}{constrained_suffix}.png"
    p_pred.save(output_path, dpi=300, verbose=False)
    print(f"  Saved unified prediction landscape: {output_path}")


def plot_delta_score_landscape(df_grid, df_datapoints, output_dir, which_space, only_x,
                                same_type_suffix=""):
    """
    Create delta score heatmap showing shift towards plural (positive) or singular (negative).

    Delta score = P_before(singular) - P_after(singular) + P_after(plural) - P_before(plural)
    - Positive: intervention shifted towards plural
    - Negative: intervention shifted towards singular
    """
    if not PLOTTING_AVAILABLE:
        return

    color_map = {
        ' is': '#0066CC',
        ' has': '#4D94FF',
        ' was': '#99CCFF',
        ' are': '#CC0000',
        ' have': '#FF4D4D',
        ' were': '#FF9999'
    }

    # Use diverging colormap: blue (singular/negative) to red (plural/positive)
    p_delta = (
        p9.ggplot(df_grid) +
        p9.aes(x='grid_x', y='grid_y', fill='delta_score') +
        p9.geom_tile() +
        p9.geom_point(
            data=df_datapoints,
            mapping=p9.aes(x='x', y='y', color='label'),
            size=1.5,
            alpha=0.6,
            inherit_aes=False
        ) +
        p9.scale_fill_gradient2(
            low='#0066CC',      # Blue for singular (negative)
            mid='white',         # White for neutral
            high='#CC0000',      # Red for plural (positive)
            midpoint=0,
            name='Delta Score'
        ) +
        p9.scale_color_manual(values=color_map) +
        p9.labs(
            x='Dimension 1',
            y='Dimension 2',
            title='Delta Score: Shift Towards Plural (+) or Singular (-)',
            color='Datapoint Label'
        ) +
        p9.coord_fixed(ratio=1) +  # Equal aspect ratio: 1 unit x = 1 unit y
        p9.theme_minimal() +
        p9.theme(
            panel_background=p9.element_rect(fill='white'),
            panel_border=p9.element_rect(color='black'),
            figure_size=(10, 8)
        ) +
        p9.scale_x_continuous(expand=(0, 0)) +
        p9.scale_y_continuous(expand=(0, 0))
    )

    output_path = f"{output_dir}/unified_delta_score_{which_space}_onlyx{only_x}{same_type_suffix}.png"
    p_delta.save(output_path, dpi=300, verbose=False)
    print(f"  Saved delta score landscape: {output_path}")


def collect_datapoint_projections(all_examples, which_space):
    """Create dataframe of datapoint locations for scatter plot overlay"""
    results = []
    for ex in all_examples:
        results.append({
            'x': ex['base_x'],
            'y': ex['base_y'],
            'label': ex['base_label_str'],
            'point_type': 'base'
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Generate unified intervention visualization")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("--intervention", type=str, default="low_rank_invertible_unconstrained")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="agr_sv_num_pp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grid_resolution", type=int, default=25)
    parser.add_argument("--only_x", action="store_true", help="Only vary X dimension in grid")
    parser.add_argument("--which_space", type=str, default="other", choices=["projection", "other"])
    parser.add_argument("--output_dir", type=str, default="./viz_debug")
    parser.add_argument("--use_eval", default=True, action="store_true", help="Use evalset instead of trainset")
    parser.add_argument("--include_same_type", action="store_true", default=False,
                        help="Include pairs with same feature types")
    parser.add_argument("--constrained_prediction", action="store_true", default=False,
                        help="Plot most likely verb instead of overall most likely token")
    parser.add_argument("--plot_by_class", action="store_true", default=False,
                        help="Plot verb regions by class (singular/plural)")
    parser.add_argument("--use_token_metric", action="store_true", default=False,
                        help="Use token-level correctness instead of feature-level")
    parser.add_argument("--intervention_folder", type=str, default="regular",
                        help="Subfolder in cached_interventions to load from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for base example selection")
    parser.add_argument("--total_samples", type=int, default=200, help="Total samples to load from dataset")
    parser.add_argument("--eval_samples", type=int, default=50, help="Eval samples to use")
    parser.add_argument("--range_scale", type=float, default=1.0,
                        help="Scale factor for x/y range (e.g., 2.0 doubles the range)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load everything
    gpt, tokenizer, intervenable, collect_gpt, rotate_layer, other_layer, interchange_dim = load_models_and_intervention(
        args.model, args.intervention, args.layer, args.position,
        low_rank_dimension=2, num_layers=None, hidden_dim=None, device=args.device,
        intervention_folder=args.intervention_folder
    )

    # Get data
    trainset, evalset, data_source = get_data(
        dataset=args.dataset,
        tokenizer=tokenizer,
        total_samples=args.total_samples,
        eval_samples=args.eval_samples,
        batch_size=4,
        device=args.device,
        seed=42,
        include_same_type=args.include_same_type
    )

    dataset = evalset if args.use_eval else trainset
    dataset_name = "evalset" if args.use_eval else "trainset"
    same_type_suffix = "_same_type" if args.include_same_type else ""

    print(f"\n{'='*60}")
    print(f"UNIFIED VISUALIZATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Intervention: {args.intervention}")
    print(f"Layer: {args.layer}, Position: {args.position}")
    print(f"Interchange dim: {interchange_dim}")
    print(f"Dataset: {args.dataset} ({dataset_name})")
    print(f"Include same type: {args.include_same_type}")
    print(f"Space: {args.which_space}, Only X: {args.only_x}")
    print(f"Grid resolution: {args.grid_resolution}")
    print(f"Random seed: {args.seed}")

    # Collect all examples
    print(f"\n{'='*60}")
    print(f"COLLECTING ALL EXAMPLES")
    print(f"{'='*60}")
    all_examples = collect_all_examples(
        collect_gpt, dataset, rotate_layer, other_layer, args.position, tokenizer,
        args.which_space, args.device
    )
    print(f"Collected {len(all_examples)} examples")

    # Determine grid range from datapoints
    all_x = [ex['base_x'] for ex in all_examples]
    all_y = [ex['base_y'] for ex in all_examples]

    # Calculate initial ranges with margin
    x_min, x_max = min(all_x) - 0.5, max(all_x) + 0.5
    y_min, y_max = min(all_y) - 0.5, max(all_y) + 0.5

    # Make both axes have the same span (square grid)
    x_span = x_max - x_min
    y_span = y_max - y_min
    max_span = max(x_span, y_span)

    # Apply range scale factor
    scaled_span = max_span * args.range_scale

    # Center each axis and extend to scaled_span
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = (x_center - scaled_span / 2, x_center + scaled_span / 2)
    y_range = (y_center - scaled_span / 2, y_center + scaled_span / 2)

    print(f"\nDatapoint statistics:")
    print(f"  X range: [{min(all_x):.2f}, {max(all_x):.2f}]")
    print(f"  Y range: [{min(all_y):.2f}, {max(all_y):.2f}]")
    print(f"\nGrid range (equalized span {max_span:.2f} x scale {args.range_scale} = {scaled_span:.2f}):")
    print(f"  X: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
    print(f"  Y: [{y_range[0]:.2f}, {y_range[1]:.2f}]")

    # Perform unified grid sweep
    print(f"\n{'='*60}")
    print(f"UNIFIED GRID SWEEP")
    print(f"{'='*60}")
    print(f"Each grid point will use a randomly selected base example")

    df_grid = unified_grid_sweep(
        gpt_model=gpt,
        all_examples=all_examples,
        rotate_layer=rotate_layer,
        other_layer=other_layer,
        intervention_layer=args.layer,
        x_range=x_range,
        y_range=y_range,
        grid_resolution=args.grid_resolution,
        only_x=args.only_x,
        which_space=args.which_space,
        device=args.device,
        tokenizer=tokenizer,
        seed=args.seed
    )

    # Save grid data
    grid_file = f"{args.output_dir}/unified_grid_{dataset_name}_{args.which_space}_onlyx{args.only_x}{same_type_suffix}.csv"
    df_grid.to_csv(grid_file, index=False)
    print(f"\nSaved grid data to: {grid_file}")

    # Create datapoints dataframe for plotting
    df_datapoints = collect_datapoint_projections(all_examples, args.which_space)

    # Generate plots
    if PLOTTING_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"GENERATING PLOTS")
        print(f"{'='*60}")
        plot_unified_landscape(
            df_grid, df_datapoints, args.output_dir, args.which_space, args.only_x,
            same_type_suffix, args.constrained_prediction, args.use_token_metric, args.plot_by_class
        )
        # Add delta score plot
        plot_delta_score_landscape(
            df_grid, df_datapoints, args.output_dir, args.which_space, args.only_x,
            same_type_suffix
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    feature_acc = df_grid['correct_feature'].mean()
    token_acc = df_grid['correct_token'].mean()
    print(f"Unified grid accuracy (feature-level): {feature_acc:.2%}")
    print(f"Unified grid accuracy (token-level): {token_acc:.2%}")

    # Delta score statistics
    print(f"\nDelta score statistics (+ = plural, - = singular):")
    print(f"  Mean: {df_grid['delta_score'].mean():.4f}")
    print(f"  Std:  {df_grid['delta_score'].std():.4f}")
    print(f"  Min:  {df_grid['delta_score'].min():.4f}")
    print(f"  Max:  {df_grid['delta_score'].max():.4f}")
    print(f"  Points shifting towards plural (delta > 0): {(df_grid['delta_score'] > 0).sum()} ({(df_grid['delta_score'] > 0).mean():.1%})")
    print(f"  Points shifting towards singular (delta < 0): {(df_grid['delta_score'] < 0).sum()} ({(df_grid['delta_score'] < 0).mean():.1%})")

    # Breakdown by predicted class
    print(f"\nPrediction distribution:")
    print(df_grid['most_likely_verb'].value_counts())

    print(f"\nBase label distribution in grid (randomly sampled):")
    print(df_grid['base_label_str'].value_counts())


if __name__ == "__main__":
    main()
