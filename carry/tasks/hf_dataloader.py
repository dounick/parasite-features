from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path
from causal.counterfactual_dataset import CounterfactualDataset
import os

def load_hf_dataset(dataset_path, split, parse_fn, hf_token=None, size=None,
                    name=None, ignore_names=[], shuffle=False, filter_fn=None):
    """
    Load a HuggingFace dataset and reformat it to be compatible with the
    CounterfactualDataset class.

    Args:
        dataset_path (str): The path or name of the HF dataset
        split (str): Dataset split to load ("train", "test", or "validation")
        hf_token (str): HuggingFace authentication token
        size (int, optional): Number of examples to load. Defaults to None (all).
        name (str, optional): Sub-configuration name for the dataset. Defaults to None.
        parse_fn (callable, optional): A function that takes a single row from a
            dataset and returns a string or dict to be placed in the "input" column.
            If None, defaults to using row["question"] or row["prompt"].
        ignore_names (list, optional): Names to ignore when looking for counterfactuals.
            Defaults to empty list.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

    Returns:
        dict: A dictionary containing CounterfactualDataset objects, one for each
                counterfactual type. Keys are formatted as "{counterfactual_name}_{split}".
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if dataset_path == "mib-bench/arithmetic_addition":
        local = Path(__file__).resolve().parents[1] / "data" / "arithmetic_addition" / split
        base_dataset = load_from_disk(str(local))
    else:
        base_dataset = load_dataset(dataset_path, name, split=split, token=hf_token)
    if filter_fn is not None:
        base_dataset = base_dataset.filter(filter_fn)

    if shuffle:
        base_dataset = base_dataset.shuffle(seed=42)

    if size is not None:
        if size > len(base_dataset):
            size = len(base_dataset)
        base_dataset = base_dataset.select(range(size))

    # Retrieve all counterfactual names
    sample = base_dataset[0]
    counterfactual_names = [
        k for k in sample.keys()
        if k.endswith('_counterfactual')
        and not any(name in k for name in ignore_names)
    ]

    data_dict = {
        counterfactual_name: {"input": [], "counterfactual_inputs": []}
        for counterfactual_name in counterfactual_names
    }

    for row in base_dataset:
        try:
            input_obj = parse_fn(row)
        except Exception as e:
            print(f"Error parsing input: {e} for row {row}")
            continue

        for counterfactual_name in counterfactual_names:
            if counterfactual_name in row:
                cf_data = row[counterfactual_name]
            else:
                cf_data = []

            data_dict[counterfactual_name]["input"].append(input_obj)
            counterfactual_obj = parse_fn(cf_data)
            if not isinstance(counterfactual_obj, list):
                counterfactual_obj = [counterfactual_obj]
            data_dict[counterfactual_name]["counterfactual_inputs"].append(
                counterfactual_obj
            )


    datasets = {}
    for counterfactual_name in data_dict.keys():
        try:
            name = counterfactual_name.replace("_counterfactual", "_" + split)
            hf_dataset = Dataset.from_dict(data_dict[counterfactual_name])
            datasets[name] = CounterfactualDataset(
                dataset=hf_dataset,
                id=f"{name}"
            )
        except Exception as e:
            print(
                f"Error creating dataset for {counterfactual_name}: {e} "
                f"{type(data_dict[counterfactual_name])} "
                f"{data_dict[counterfactual_name]['input'][0]} "
                f"{data_dict[counterfactual_name]['counterfactual_inputs'][0]} "
                f"{split}"
            )
            assert False

    return datasets
