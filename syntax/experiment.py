from tempfile import template
import torch, os, argparse, datetime, pandas as pd
os.environ.setdefault("WANDB_SILENT", "true")
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyvene import set_seed
from tqdm import tqdm

from interventions.das import run_das_experiment
from utils import DEFAULT_CONFIG, neural_interventions, get_last_token
from data.data import Dataset


def eval_behavior(model, evalset, device):
    output = []
    for batch in tqdm(evalset, desc="Evaluating factual behavior"):
        base_labels = batch.base_labels
        source_labels = batch.src_labels

        base_logits = get_last_token(model(**batch.base).logits, batch.base['attention_mask'])
        base_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
        base_base_probs = torch.gather(base_probs, 1, base_labels.unsqueeze(-1)).squeeze(-1)
        base_source_probs = torch.gather(base_probs, 1, source_labels.unsqueeze(-1)).squeeze(-1)

        # source_probs = embed_to_distrib(model, model(**batch.src).logits[:, -1, :], log=True)
        # source_base_probs = torch.gather(source_probs, 1, base_labels.unsqueeze(-1)).squeeze(-1)
        # source_source_probs = torch.gather(source_probs, 1, source_labels.unsqueeze(-1)).squeeze(-1)

        correctness = (base_base_probs < base_source_probs)
        output.append(correctness.cpu())

    accuracy = torch.cat(output).float().mean().item()
    return accuracy


def get_data(dataset, tokenizer, total_samples, eval_samples, batch_size, device, manipulate=None, seed=42, include_same_type=False):
    steps = total_samples // batch_size
    data_source = Dataset.load_from(f'templates/{dataset}', bos_token=tokenizer.bos_token or '<|endoftext|>')
    trainset = data_source.sample_batches(tokenizer, batch_size, steps, device, seed=seed, manipulate=manipulate, include_same_type=include_same_type)

    discard = set()
    for batch in trainset:
        for pair in batch.pairs:
            discard.add(''.join(pair.base))

    evalset = data_source.sample_batches(tokenizer, batch_size, eval_samples // batch_size, device, seed=seed, discard=discard, manipulate=manipulate, include_same_type=include_same_type)
    return trainset, evalset, data_source


def main(model, dataset, stream, total_samples, batch_size, epochs, low_rank_dimension, intervention_dim, learning_rate, gradient_accumulation_steps, \
         intervention_types, eval_factual, num_layers, hidden_dim, seed, cache_dir, device, wandb, save_intervention=False, changed_params={}, include_same_type=False, layer=None, position=None):

    # Set seed for reproducibility
    set_seed(seed)

    # Create output directory with intervention type
    model_name = model.split("/")[-1].lower()
    output_dir = f"results/"

    print(f"Loading model {model}...")
    # Load model and tokenizer matching the notebook
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(model)#, cache_dir = '../../../shared/hf_cache')
    _ = gpt.to(device)
    _ = gpt.eval()  # always no grad on the model

    # Create datasets
    trainset, evalset, data_source = get_data(
        dataset = dataset,
        tokenizer = tokenizer,
        total_samples = total_samples,
        eval_samples = 50,
        batch_size = batch_size,
        device = device,
        seed = seed,
        include_same_type = include_same_type
    )

    if eval_factual or not DEFAULT_CONFIG["baseline_accuracy"]:
        factual_acc = eval_behavior(gpt, evalset, device)
        print(f"Factual accuracy: {factual_acc:.4f}")
    else:
        factual_acc = DEFAULT_CONFIG["baseline_accuracy"]  # Use predefined baseline if available


    _ = gpt.to("cpu")
    del gpt
    gpt = AutoModelForCausalLM.from_pretrained(model)
    _ = gpt.to(device)
    _ = gpt.eval()  # always no grad on the model


    NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output=[]

    # Determine position and layer ranges
    position_range = [position] if position is not None else range(data_source.first_var_pos, data_source.length)
    layer_range = [layer] if layer is not None else range(gpt.config.num_hidden_layers)

    for intervention_type in tqdm(intervention_types, desc="Intervention Types", position=0, leave=True):
        for pos_i in tqdm(position_range, desc="Positions", position=1, leave=False):
            for layer_i in tqdm(layer_range, desc="Layers", position=2, leave=False):
                extra_id = f'_nlayers{num_layers}_hiddendim{hidden_dim}' if intervention_type in neural_interventions else ''
                result = run_das_experiment(
                    intervention_name=intervention_type,
                    model=gpt,
                    layer=layer_i,
                    component=stream,
                    trainset=trainset,
                    evalset=evalset,
                    n_train=total_samples,
                    intervention_pos=pos_i,
                    epochs=epochs,
                    low_rank_dimension=low_rank_dimension,
                    intervention_dim=intervention_dim,
                    include_same_type=include_same_type,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    device=device,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim
                    ,
                    # Pass wandb settings via config in run_das_experiment
                    config={
                        "use_wandb": wandb,
                        "wandb_project": 'expressivity',
                        "wandb_run_name": f"{dataset}_{intervention_type}_layer{layer_i}{extra_id}_position_{pos_i}_seed{seed}",
                        "wandb_run_id": f"{dataset}_{intervention_type}_layer{layer_i}{extra_id}_position_{pos_i}_seed{seed}_{NOW}",
                        "tag": dataset
                    },
                    save_intervention=save_intervention,
                )
                output.append(result)

        output_df = pd.DataFrame(output)
        os.makedirs(output_dir+f'/raw/{model_name}/{dataset}', exist_ok=True)
        changed_params_extra = '_' + '_'.join([f"{key}{value}" for key, value in changed_params.items()]) if changed_params else ''
        out_path = os.path.join(output_dir+f'/raw/{model_name}/{dataset}{changed_params_extra}')
        os.makedirs(out_path, exist_ok=True)
        output_path = os.path.join(out_path, f"{intervention_type}_{NOW}.csv")
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DAS Experiment")

    # Basic arguments
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1b",
                        help="Model to use for training")
    parser.add_argument("--intervention_types", type=str, nargs='+', default=['low_rank'],
                        choices=["das", "low_rank", "low_rank_custom_dim", "boundless", "stretch", "realnvp", "revnet", "unconstrained", "low_rank_invertible_unconstrained", "low_rank_invertible_unconstrained_2d", "low_rank_invertible_realnvp", "low_rank_invertible_revnet", "low_rank_invertible_rotated"],
                        help="Type of intervention to apply")
    parser.add_argument("--template", type=str, nargs='+', default = None,
                        choices = ["agr_sv_num_pp", "agr_sv_num_pp_animacy_parasite", "agr_sv_num_pp_gender_parasite", "agr_sv_num_num_parasite", "agr_sv_num_plus10_parasite", "agr_sv_num_n_parasite", "agr_sv_num_caps_parasite"],
                        help="Template name for the dataset")

    # Training parameters
    parser.add_argument("--stream", type=str, default="block_output")
    parser.add_argument("--low_rank_dimension", type=int, default=DEFAULT_CONFIG["low_rank_dimension"],
                        help="Low rank dimension for low_rank intervention")
    parser.add_argument("--intervention_dim", type=int, default=1,
                        help="Number of dimensions to intervene on (for low_rank_custom_dim, must be <= low_rank_dimension)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--total_samples", type=int, default=DEFAULT_CONFIG["total_samples"],
                        help="Total number of samples to generate")
    parser.add_argument("--include_same_type", action="store_true", default=False,
                        help="Include pairs with same feature types in addition to different types (doubles dataset size)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers for the intervention")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for the intervention")
    parser.add_argument("--layer", type=int, default=None, help="Specific layer to run (if not provided, runs all layers)")
    parser.add_argument("--position", type=int, default=None, help="Specific position to run (if not provided, runs all positions)")

    # Optional wandb integration
    parser.add_argument("--wandb", "-wb", action="store_true", default=False, help="Enable logging to Weights & Biases")

    # Misc parameters
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save results")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CONFIG["cache_dir"],
                        help="Cache directory for models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_factual", action="store_true",
                        help="Evaluate factual performance first")
    parser.add_argument("--save_intervention", "-s", action="store_true",
                        help="Whether to save the learned intervention models")

    args = parser.parse_args()
    args.wandb = not args.wandb # Flag should turn it off

    # Identify changed parameters from defaults
    changed_params = {}
    for key, value in vars(args).items():
        if key in DEFAULT_CONFIG and value != DEFAULT_CONFIG[key]:
            changed_params[key] = value

    for template in args.template:
        main(
            model=args.model,
            dataset=template,
            stream='block_output',
            total_samples=args.total_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            low_rank_dimension=args.low_rank_dimension,
            intervention_dim=args.intervention_dim,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            intervention_types=args.intervention_types,
            eval_factual=args.eval_factual,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            cache_dir=args.cache_dir,
            device=args.device,
            wandb=args.wandb,
            save_intervention=args.save_intervention,
            changed_params=changed_params,
            include_same_type=args.include_same_type,
            layer=args.layer,
            position=args.position
        )
