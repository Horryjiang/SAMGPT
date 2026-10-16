"""SAMGPT full-graph experiment: pretraining -> prompt adaptation -> evaluation.

Local corrections are documented in gfm_samgpt/docs/05_原仓库本地修复.md.
Importing this module or running --help does not start an experiment.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from runtime import ROOT, DATA_DIR, choose_device, fewshot_dir, resolve_path


def build_parser():
    parser = argparse.ArgumentParser('SAMGPT')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--pretrain_datasets', nargs='+', default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'])
    parser.add_argument('--downstream_task', choices=['node', 'graph'], default='node')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:N')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index for auto/cuda; retained from upstream.')
    parser.add_argument('--pretrain_method', choices=['GRAPHCL', 'LP', 'splitLP'], default='GRAPHCL')
    parser.add_argument('--aug_type', choices=['edge'], default='edge', help='Only edge augmentation is wired into upstream PrePrompt.')
    parser.add_argument('--drop_percent', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--combinetype', choices=['mul', 'add'], default='mul')
    parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='1-based source graph IDs, applied to each source dataset.')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--negative_samples_num', type=int, default=40)
    parser.add_argument('--skip_pretrain', type=int, choices=[0, 1], default=1, help='1: reuse compatible checkpoint if present; 0: train, refusing checkpoint overwrite.')
    parser.add_argument('--ablation_pre', choices=['all', 'None', 'st', 'ft'], default='all')
    parser.add_argument('--ablation_down', default='all')
    parser.add_argument('--unify_dim', type=int, default=50)
    parser.add_argument('--shot_num', type=int, default=1, help='Evaluate shots 1 through shot_num.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--layers_num', type=int, default=3)
    parser.add_argument('--backbone', choices=['gcn', 'gat'], default='gcn')
    parser.add_argument('--pretrain-epochs', type=int, default=10000)
    parser.add_argument('--adapt-epochs', type=int, default=400)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--data-dir', type=resolve_path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=resolve_path, default=ROOT)
    parser.add_argument('--checkpoint', type=resolve_path, help='Explicit compatible local checkpoint path.')
    parser.add_argument('--pretrain-only', action='store_true', help='Stop after saving/loading the pretrained model.')
    return parser


def pretrain_config(args):
    keys = ['pretrain_datasets', 'graphId', 'pretrain_method', 'aug_type', 'drop_percent', 'seed',
            'combinetype', 'alpha', 'negative_samples_num', 'ablation_pre', 'unify_dim',
            'lr', 'hid_units', 'layers_num', 'backbone', 'pretrain_epochs', 'patience']
    return dict(revision='local-v1', data_dir=str(args.data_dir), **{key: getattr(args, key) for key in keys})


def load_support(args, shot, episode, labels):
    import torch
    folder = fewshot_dir(args.data_dir, args.dataset, shot, episode, args.downstream_task)
    keys = ['idx', 'labels', 'centers', 'test_idx']
    if args.downstream_task == 'graph':
        keys.append('batch')
    for key in keys:
        if not (folder / f'{key}.pt').is_file():
            raise FileNotFoundError(f'Missing {folder / (key + ".pt")}. Generate matching shots/episodes/task/test-count with src/utils/generate_fewshot.py first; old splits must be regenerated in a new data directory.')
    sample = {key: torch.load(folder / f'{key}.pt', map_location='cpu', weights_only=True).long().reshape(-1) for key in keys}
    centers = sample['centers']
    expected_test = torch.arange(len(labels) - args.test_count, len(labels))
    if not torch.equal(sample['test_idx'], expected_test):
        raise ValueError(f'{folder}: test split differs from --test-count.')
    classes = torch.unique(labels)
    if not torch.equal(classes, torch.arange(len(classes))):
        raise ValueError('Expected contiguous node class labels 0, ..., C-1.')
    if len(centers) != len(classes) * shot or len(centers.unique()) != len(centers):
        raise ValueError(f'{folder}: invalid support size or duplicate centers.')
    if (centers < 0).any() or (centers >= len(labels) - args.test_count).any():
        raise ValueError(f'{folder}: support centers overlap test nodes or are out of range.')
    if not torch.equal(labels[centers], sample['labels']) or not torch.equal(torch.bincount(sample['labels'], minlength=len(classes)), torch.full((len(classes),), shot)):
        raise ValueError(f'{folder}: labels or per-class shot counts do not match the dataset.')
    if (sample['idx'] < 0).any() or (sample['idx'] >= len(labels)).any():
        raise ValueError(f'{folder}: invalid node indices.')
    if args.downstream_task == 'node' and not torch.equal(sample['idx'], centers):
        raise ValueError(f'{folder}: node indices differ from support centers.')
    if args.downstream_task == 'graph':
        batch = sample['batch']
        if len(batch) != len(sample['idx']) or not torch.equal(batch.unique(), torch.arange(len(centers))):
            raise ValueError(f'{folder}: invalid subgraph batch indices.')
        for i, center in enumerate(centers):
            if center not in sample['idx'][batch == i]:
                raise ValueError(f'{folder}: missing subgraph center.')
    return sample


def pretrain(args, model, device, checkpoint, config):
    import scipy.sparse as sp
    import torch
    from preprompt import pca_compression, prompt_pretrain_sample
    from utils.dataset import load_dataset
    from utils import aug, process
    if checkpoint.exists():
        if not args.skip_pretrain:
            raise FileExistsError(f'Refusing to overwrite {checkpoint}; use a new --checkpoint or --output-dir.')
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get('config') != config:
            raise ValueError(f'Incompatible checkpoint: {checkpoint}. Select a new path and retrain.')
        model.load_state_dict(saved['model'])
        print(f'Loaded checkpoint: {checkpoint}')
        return
    features, adjs, aug_features, aug_adjs, labels, samples = [], [], [], [], [], []
    # No legacy cache reuse: old files did not encode PCA/seed/augmentation settings.
    for name in args.pretrain_datasets:
        dataset = load_dataset(name, args.data_dir)
        for graph_id in args.graphId:
            if graph_id > len(dataset):
                raise ValueError(f'{name} has {len(dataset)} graphs, requested graphId={graph_id}.')
            data = dataset[graph_id - 1]
            feature, adj = process.process_tu(data, data.x.shape[1])
            feature = torch.as_tensor(pca_compression(feature, args.unify_dim), dtype=torch.float32)
            if args.pretrain_method == 'GRAPHCL':
                af, aa, label = aug.build_aug(adj, feature, True, args.drop_percent)
                aug_features.append(af.to(device)); aug_adjs.append(aa.to(device)); labels.append(label.to(device))
            elif args.pretrain_method == 'splitLP':
                samples.append(prompt_pretrain_sample(adj, args.negative_samples_num).to(device))
            features.append(feature.to(device))
            adjs.append(process.normalize_adj(adj + sp.eye(adj.shape[0])))
    if args.pretrain_method == 'LP':
        samples = prompt_pretrain_sample(process.combine_dataset_list_sp(adjs), args.negative_samples_num).to(device)
    adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).to(device) for adj in adjs]
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, wait = float('inf'), 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.pretrain_epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, True, None, None, None, labels)
        else:
            loss = model(features, adjs, True, None, None, None, None, samples=samples)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite pretraining loss at epoch {epoch + 1}.')
        value = loss.item()
        loss.backward()
        optimiser.step()
        # As in upstream, loss selects this epoch's post-update checkpoint.
        if value < best:
            best, wait = value, 0
            torch.save({'config': config, 'model': model.state_dict(), 'epoch': epoch + 1, 'loss': best}, checkpoint)
        else:
            wait += 1
        print(f'Pretrain {epoch + 1}/{args.pretrain_epochs}: loss={value:.6f}')
        if wait >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model'])
    print(f'Best checkpoint: {checkpoint}')


def evaluate(args, model, device, data, support_sets, run_dir):
    import numpy as np
    import scipy.sparse as sp
    import torch
    from sklearn.metrics import f1_score
    from preprompt import pca_compression
    from downprompt import downprompt, downprompt_graph
    from utils import process
    # Reset for identical adaptation whether pretraining was run or loaded.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model.eval().requires_grad_(False)
    feature, raw_adj = process.process_tu(data, data.x.shape[1])
    feature = torch.as_tensor(pca_compression(feature, args.unify_dim), dtype=torch.float32, device=device)
    adj = process.sparse_mx_to_torch_sparse_tensor(process.normalize_adj(raw_adj + sp.eye(raw_adj.shape[0]))).to(device)
    test_idx = torch.arange(len(data.y) - args.test_count, len(data.y))
    test_labels = data.y[test_idx].to(device)
    test_args = [test_idx.to(device)]
    if args.downstream_task == 'graph':
        subgraph = process.build_subgraph(raw_adj, test_idx, sparse=True)
        test_args = [subgraph['idx'].to(device), subgraph['batch'].to(device)]
    head_class = downprompt_graph if args.downstream_task == 'graph' else downprompt
    rows = []
    for shot in range(1, args.shot_num + 1):
        metrics = []
        for episode in range(args.episodes):
            sample = support_sets[shot, episode]
            idx, labels = sample['idx'].to(device), sample['labels'].to(device)
            train_args = [idx]
            if args.downstream_task == 'graph':
                train_args.append(sample['batch'].to(device))
            fea, structure, combines = model.get_weights()
            log = head_class(args.hid_units, len(torch.unique(data.y)), args.unify_dim, args.layers_num,
                             fea, structure, combines + [args.beta], args.combinetype, args.ablation_down).to(device)
            opt = torch.optim.Adam(log.parameters(), lr=0.001)
            best, wait, best_state = float('inf'), 0, None
            for epoch in range(args.adapt_epochs):
                log.train(); opt.zero_grad(set_to_none=True)
                logits = log(feature, adj, True, model.gcn, *train_args, labels=labels, train=1, return_logits=True)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite adaptation loss.')
                if loss.requires_grad:
                    loss.backward()
                    opt.step()
                if loss.item() < best:
                    best, wait = loss.item(), 0
                    best_state = {key: value.detach().clone() for key, value in log.state_dict().items()}
                else:
                    wait += 1
                if wait >= args.patience or not loss.requires_grad:
                    break
            log.load_state_dict(best_state)
            log.eval()
            with torch.no_grad():
                # Refresh prototypes from support labels using the restored prompt weights.
                log(feature, adj, True, model.gcn, *train_args, labels=labels, train=1)
                logits = log(feature, adj, True, model.gcn, *test_args, return_logits=True)
                preds = logits.argmax(dim=1)
                accuracy = (preds == test_labels).float().mean().item() * 100
                truth, predictions = test_labels.cpu().numpy(), preds.cpu().numpy()
                micro = f1_score(truth, predictions, average='micro') * 100
                macro = f1_score(truth, predictions, average='macro', zero_division=0) * 100
            metrics.append([accuracy, micro, macro])
            print(f'{shot}-shot episode {episode + 1}/{args.episodes}: accuracy={accuracy:.2f}%')
        values = np.asarray(metrics)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1 if args.episodes > 1 else 0)
        rows.append([args.dataset, shot, args.episodes, *mean.tolist(), *std.tolist()])
    with (run_dir / 'metrics.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'shot', 'episodes', 'accuracy_mean', 'micro_f1_mean', 'macro_f1_mean', 'accuracy_std', 'micro_f1_std', 'macro_f1_std'])
        writer.writerows(rows)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = ['unify_dim', 'hid_units', 'layers_num', 'shot_num', 'pretrain_epochs', 'adapt_epochs', 'episodes', 'patience', 'test_count', 'negative_samples_num']
    if any(getattr(args, key) < 1 for key in positive) or args.lr <= 0:
        parser.error('Dimensions, epochs, episodes, shots, counts and lr must be positive.')
    if not 0 <= args.drop_percent <= 1 or min(args.graphId) < 1 or len(set(args.graphId)) != len(args.graphId):
        parser.error('Invalid drop_percent or graphId.')
    import numpy as np
    import torch
    from preprompt import PrePrompt
    device = choose_device(args.device, args.gpu)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = pretrain_config(args)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    checkpoint = args.checkpoint or args.output_dir / 'checkpoints' / 'local_v1' / f'{digest}.pt'
    support_sets = {}
    if not args.pretrain_only:
        from utils.dataset import load_dataset
        dataset = load_dataset(args.dataset, args.data_dir)
        if len(dataset) != 1 or dataset[0].y.ndim != 1:
            raise ValueError('Downstream supports single-graph node / ego-subgraph datasets with 1-D labels.')
        data = dataset[0]
        if not 0 < args.test_count < len(data.y):
            parser.error('test-count must be smaller than number of target nodes.')
        # Check all support files before spending time on pretraining.
        for shot in range(1, args.shot_num + 1):
            for episode in range(args.episodes):
                support_sets[shot, episode] = load_support(args, shot, episode, data.y.cpu())
    run_dir = args.output_dir / 'result' / 'local_v1' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(vars(args), device=str(device), checkpoint=str(checkpoint), pretrain_config=config,
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
    (run_dir / 'config.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print(f'Device: {device}\nData: {args.data_dir}\nResults: {run_dir}')
    model = PrePrompt(args.unify_dim, args.hid_units, 'prelu', len(args.pretrain_datasets) * len(args.graphId),
                      args.layers_num, 0.1, type_=args.combinetype, backbone=args.backbone,
                      alpha=args.alpha, ablation=args.ablation_pre).to(device)
    pretrain(args, model, device, checkpoint, config)
    if not args.pretrain_only:
        evaluate(args, model, device, data, support_sets, run_dir)
    print(f'Finished. Results: {run_dir}')


if __name__ == '__main__':
    main()
