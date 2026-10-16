"""Generate reproducible support sets; keep the final test nodes out of support.

Run directly with ``python src/utils/generate_fewshot.py --help`` or, from src,
with ``python -m utils.generate_fewshot --help``.
"""
import argparse
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import DATA_DIR, fewshot_dir, resolve_path


def generate_fewshot_samples(labels, num_shots, num_samples, test_count=100, seed=39):
    import torch
    if labels.ndim != 1 or not 0 < test_count < len(labels):
        raise ValueError('Expected 1-D node labels and 0 < test_count < number of nodes.')
    if num_shots < 1 or num_samples < 1:
        raise ValueError('shots and episodes must be positive.')
    classes = torch.unique(labels)
    if not torch.equal(classes, torch.arange(len(classes))):
        raise ValueError('Expected contiguous class labels 0, ..., C-1.')
    pools = [(labels[:-test_count] == label).nonzero(as_tuple=True)[0].tolist() for label in classes]
    if any(len(pool) < num_shots for pool in pools):
        raise ValueError('A class has too few non-test nodes for the requested shots.')
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        idx = torch.tensor([i for pool in pools for i in rng.sample(pool, num_shots)])
        samples.append({'idx': idx, 'labels': labels[idx], 'centers': idx,
                        'test_idx': torch.arange(len(labels) - test_count, len(labels))})
    return samples


def save_sample(data, path):
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for key, value in data.items():
        torch.save(value, path / f'{key}.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['Cora'])
    parser.add_argument('--task', choices=['node', 'graph', 'both'], default='node')
    parser.add_argument('--shots', type=int, default=1, help='Generate 1 through this many shots.')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--data-dir', type=resolve_path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if min(args.shots, args.episodes, args.test_count) < 1:
        parser.error('shots, episodes and test-count must be positive')
    tasks = ['node', 'graph'] if args.task == 'both' else [args.task]
    # Refuse overwrite before downloading data or creating any episode directories.
    for name in args.datasets:
        for shot in range(1, args.shots + 1):
            for episode in range(args.episodes):
                for task in tasks:
                    path = fewshot_dir(args.data_dir, name, shot, episode, task)
                    if path.exists():
                        parser.error(f'Already exists: {path}. Use a new --data-dir to preserve this split.')
    from utils.dataset import load_dataset
    from utils import process
    for name in args.datasets:
        dataset = load_dataset(name, args.data_dir)
        if len(dataset) != 1:
            raise ValueError('This entry point supports single-graph node / ego-subgraph datasets.')
        data = dataset[0]
        if 'graph' in tasks:
            _, adj = process.process_tu(data, data.x.shape[1])
        for shot in range(1, args.shots + 1):
            samples = generate_fewshot_samples(data.y.cpu(), shot, args.episodes, args.test_count, args.seed)
            for episode, sample in enumerate(samples):
                for task in tasks:
                    result = dict(sample)
                    if task == 'graph':
                        result.update(process.build_subgraph(adj, sample['centers'], sparse=True))
                    save_sample(result, fewshot_dir(args.data_dir, name, shot, episode, task))
        print(f'{name}: saved {tasks}, shots=1..{args.shots}, episodes={args.episodes}, seed={args.seed}, root={args.data_dir}')


if __name__ == '__main__':
    main()
