"""Paths and device handling shared by the two original entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'


def resolve_path(value):
    # Explicit relative paths are relative to the user's working directory.
    return Path(value).expanduser().resolve()


def choose_device(name='auto', gpu=0):
    import torch
    if name == 'auto':
        name = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    elif name == 'cuda':
        name = f'cuda:{gpu}'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Python environment; check PyTorch or use --device cpu.')
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('Use --device auto, cpu, cuda or cuda:N.')
    return device


def fewshot_dir(data_dir, dataset, shot, episode, task='node'):
    name = dataset.lower()
    suffix = '_graph' if task == 'graph' else ''
    return Path(data_dir) / f'fewshot_{name}{suffix}' / f'{shot}-shot_{name}' / str(episode)
