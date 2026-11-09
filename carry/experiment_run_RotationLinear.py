import argparse
parser = argparse.ArgumentParser()
parser.add_argument('task', choices=['Carry-One'])
parser.add_argument('epochs', type=int)
parser.add_argument('--shift', type=int, choices=range(1, 11), default=10)
parser.add_argument('--layer', type=int, default=18)
parser.add_argument('--device', default='cuda:2')
parser.add_argument('--method', choices=['DAS', 'DASRotationLinear_SufficientExperiment'], default='DASRotationLinear_SufficientExperiment')
parser.add_argument('--output', default='.')
args = parser.parse_args()
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent.parent))
from tasks.two_digit_addition_task.arithmetic import get_token_positions, get_counterfactual_datasets, get_causal_model
from experiments.aggregate_experiments import residual_stream_baselines
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
import gc
import torch
import os
from tqdm import tqdm
from CausalAbstraction.causal.counterfactual_dataset import CounterfactualDataset
import copy
import pyvene as pv
gc.collect()
torch.cuda.empty_cache()
counterfactual_datasets = get_counterfactual_datasets(hf=True, size=None, load_private_data=False)
causal_model = get_causal_model(shift=args.shift)
print('CARRY SHIFT:', args.shift, flush=True)
print('Available datasets:', counterfactual_datasets.keys())
device = args.device

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def checker(output_text, expected):
    import re
    numbers_in_output = re.findall('\\d+', output_text)
    if not numbers_in_output:
        return False
    first_number = numbers_in_output[0]
    if expected[0] == '0':
        expected_no_leading_zero = expected[1:]
        return first_number == expected_no_leading_zero or first_number == expected
    return first_number == expected
model_name = 'meta-llama/Llama-3.1-8B'
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16)
pipeline.tokenizer.padding_side = 'left'
print('DEVICE:', pipeline.model.device)
sampled_example = next(iter(counterfactual_datasets.values()))[0]
print('INPUT:', sampled_example['input'])
print('EXPECTED OUTPUT:', causal_model.run_forward(sampled_example['input'])['raw_output'])
print('MODEL PREDICTION:', pipeline.dump(pipeline.generate(sampled_example['input'])))
print('\nFiltering datasets based on model performance...')
exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=1024)
token_positions = get_token_positions(pipeline, causal_model)
print('*' * 100)
print(token_positions)
token_positions = [token_positions[1]]
print(token_positions)
print('\nToken positions highlighted in samples:')
for dataset in filtered_datasets.values():
    for token_position in token_positions:
        example = dataset[0]
        print(token_position.highlight_selected_token(example['counterfactual_inputs'][0]))
        break
    break
gc.collect()
torch.cuda.empty_cache()
filtered_datasets
token_positions[0]
token_positions[0].indexer
import os
start = args.layer
end = args.layer + 1
config = {'batch_size': 64, 'evaluation_batch_size': 64, 'training_epoch': args.epochs, 'n_features': 1, 'regularization_coefficient': 0.0, 'output_scores': False, 'init_lr': 0.0001, 'rotation_dims': 2}
names = ['random']
train_data = {}
test_data = {}
for name in names:
    if name + '_train' in filtered_datasets:
        train_data[name + '_train'] = filtered_datasets[name + '_train']
    if name + '_test' in filtered_datasets:
        test_data[name + '_test'] = filtered_datasets[name + '_test']
    if name + '_testprivate' in filtered_datasets:
        test_data[name + '_testprivate'] = filtered_datasets[name + '_testprivate']
verbose = False
results_dir = str(Path(args.output) / 'mock_submission_results')
model_dir = str(Path(args.output) / 'mock_submission' / '4_arithmentic_Llama3_1_8B_modulo_carry')
target_variables = ['modulo_carry']
print('Running DAS method...')
experiment = residual_stream_baselines(pipeline=pipeline, task=causal_model, token_positions=token_positions, train_data=train_data, test_data=test_data, config=config, target_variables=target_variables, checker=checker, start=start, end=end, verbose=verbose, model_dir=model_dir, results_dir=results_dir, methods=[args.method])
clear_memory()
from experiments.residual_stream_experiment import PatchResidualStream
print('Loading trained models and running inference...')
for method in [args.method]:
    print(f'Testing {method} method...')
    config['method_name'] = method
    experiment = PatchResidualStream(pipeline, causal_model, list(range(start, end)), token_positions, checker, config=config)
    method_model_dir = os.path.join(model_dir, f"{method}_{pipeline.model.__class__.__name__}_{'-'.join(target_variables)}")
    experiment.load_featurizers(method_model_dir)
    raw_results = experiment.perform_interventions(test_data, verbose=verbose, target_variables_list=[target_variables], save_dir=results_dir + '_loaded')
    del experiment, raw_results
    clear_memory()
print('Inference completed!')
