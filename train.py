import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool

from prepare import (
    BEST_CHECKPOINT_NAME,
    MAX_PARAMS,
    evaluate_loader,
    get_feature_dims,
    get_num_tasks,
    get_train_val_loaders,
    run_validation_experiment,
)


def params_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CategoricalEncoder(nn.Module):
    def __init__(self, feature_dims, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(int(dim), emb_dim) for dim in feature_dims
        )
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values):
        encoded = 0
        for column_idx, embedding in enumerate(self.embeddings):
            encoded = encoded + embedding(values[:, column_idx])
        return encoded


class GCNBaseline(nn.Module):
    def __init__(self, node_feature_dims, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.node_encoder = CategoricalEncoder(node_feature_dims, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        x = self.node_encoder(batch.x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_repr = global_mean_pool(x, batch.batch)
        return self.readout(graph_repr)


def build_args(candidate=None):
    args = SimpleNamespace()
    args.nlayers = 3
    args.nhid = 128
    args.drop = 0.10
    args.lr = 0.001
    args.weight_decay = 0.0
    args.epochs = 400
    args.patience = 20
    args.batch_size = 200
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if candidate is not None:
        for key, value in candidate.items():
            if key != 'name':
                setattr(args, key, value)
    return args


def candidate_settings():
    return [
        {
            'name': 'gcn-baseline-L3-W128-D0.10-B200-LR0.001',
            'nlayers': 3,
            'nhid': 128,
            'drop': 0.10,
            'lr': 0.001,
            'batch_size': 200,
        }
    ]


def build_model_for_hidden_dim(args, hidden_dim):
    return GCNBaseline(
        node_feature_dims=get_feature_dims(),
        hidden_dim=hidden_dim,
        out_dim=get_num_tasks(),
        num_layers=args.nlayers,
        dropout=args.drop,
    )


def build_model(args):
    hidden_dim = int(args.nhid)
    while hidden_dim >= 8:
        model = build_model_for_hidden_dim(args, hidden_dim)
        if params_count(model) <= MAX_PARAMS:
            return model, hidden_dim
        hidden_dim -= 1
    raise RuntimeError('Unable to fit baseline model under MAX_PARAMS.')


def checkpoint_path(artifact_dir):
    return os.path.join(artifact_dir, BEST_CHECKPOINT_NAME)


def model_state_to_cpu(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(artifact_dir, payload):
    final_path = checkpoint_path(artifact_dir)
    tmp_path = final_path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def maybe_save_best_checkpoint(
    model,
    candidate,
    args,
    hidden_dim,
    params_k,
    best_record,
    artifact_dir,
    val_ap,
    val_loss,
    best_epoch,
):
    is_better = (
        (val_ap > best_record['val_ap']) or
        (val_ap == best_record['val_ap'] and val_loss < best_record['val_loss'])
    )
    if not os.path.exists(checkpoint_path(artifact_dir)):
        is_better = True
    if not is_better:
        return best_record, False

    payload = {
        'candidate': dict(candidate),
        'hidden_dim': int(hidden_dim),
        'params_k': float(params_k),
        'batch_size': int(args.batch_size),
        'val_ap': float(val_ap),
        'val_loss': float(val_loss),
        'best_epoch': int(best_epoch),
        'model_state': model_state_to_cpu(model),
    }
    save_checkpoint(artifact_dir, payload)
    best_record = {
        'val_ap': float(val_ap),
        'val_loss': float(val_loss),
        'candidate': dict(candidate),
        'params_k': float(params_k),
    }
    return best_record, True


def train_candidate(candidate, budget_seconds, artifact_dir, best_record):
    args = build_args(candidate)
    train_loader, val_loader = get_train_val_loaders(args.batch_size)
    model, hidden_dim = build_model(args)
    params_k = params_count(model) / 1000.0
    model = model.to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_train_loss = float('inf')
    bad_counter = 0
    not_improved = 0
    candidate_best_val_ap = float('-inf')
    candidate_best_val_loss = float('inf')
    start_time = time.perf_counter()

    best_record, _ = maybe_save_best_checkpoint(
        model=model,
        candidate=candidate,
        args=args,
        hidden_dim=hidden_dim,
        params_k=params_k,
        best_record=best_record,
        artifact_dir=artifact_dir,
        val_ap=float('-inf'),
        val_loss=float('inf'),
        best_epoch=-1,
    )

    for epoch in range(args.epochs):
        if time.perf_counter() - start_time >= budget_seconds:
            break

        model.train()
        total_train_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            if time.perf_counter() - start_time >= budget_seconds:
                break

            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            targets = batch.y.view(logits.size(0), -1).float()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            num_graphs = batch.num_graphs
            total_train_loss += loss.item() * num_graphs
            total_graphs += num_graphs

        if total_graphs == 0:
            break

        train_loss = total_train_loss / total_graphs
        val_ap, val_loss = evaluate_loader(model, val_loader, args.device)

        if (val_ap > candidate_best_val_ap) or (
            val_ap == candidate_best_val_ap and val_loss < candidate_best_val_loss
        ):
            candidate_best_val_ap = val_ap
            candidate_best_val_loss = val_loss
            best_record, saved = maybe_save_best_checkpoint(
                model=model,
                candidate=candidate,
                args=args,
                hidden_dim=hidden_dim,
                params_k=params_k,
                best_record=best_record,
                artifact_dir=artifact_dir,
                val_ap=val_ap,
                val_loss=val_loss,
                best_epoch=epoch,
            )
            not_improved = 0
            if saved:
                print(
                    f"Checkpointed {candidate['name']} "
                    f"val_ap={val_ap:.3f} epoch={epoch} params_k={params_k:.1f}",
                    flush=True,
                )
        else:
            not_improved += 1

        if train_loss > best_train_loss:
            bad_counter += 1
        else:
            best_train_loss = train_loss
            bad_counter = 0

        if bad_counter >= 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] / 3.0, 1e-4)
            bad_counter = 0

        if not_improved >= args.patience:
            break

    print(
        f"Candidate done: {candidate['name']} "
        f"best_val_ap={candidate_best_val_ap:.3f} params_k={params_k:.1f}",
        flush=True,
    )
    return best_record


def run_training_session(artifact_dir, budget_seconds):
    set_seed(42)
    candidates = list(candidate_settings())
    best_record = {
        'val_ap': float('-inf'),
        'val_loss': float('inf'),
        'candidate': None,
        'params_k': 0.0,
    }

    session_start = time.perf_counter()
    for index, candidate in enumerate(candidates):
        elapsed = time.perf_counter() - session_start
        remaining = budget_seconds - elapsed
        if remaining <= 0:
            break

        remaining_candidates = len(candidates) - index
        candidate_budget = remaining / remaining_candidates
        print(f"Running {candidate['name']} with budget {candidate_budget:.1f}s", flush=True)
        best_record = train_candidate(
            candidate=candidate,
            budget_seconds=candidate_budget,
            artifact_dir=artifact_dir,
            best_record=best_record,
        )


def build_worker_command(artifact_dir, budget_seconds):
    return [
        sys.executable,
        os.path.abspath(__file__),
        '--worker-dir',
        artifact_dir,
        '--budget',
        str(float(budget_seconds)),
    ]


def load_checkpoint_model(checkpoint_file):
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    candidate = dict(checkpoint['candidate'])
    candidate['nhid'] = int(checkpoint['hidden_dim'])

    args = build_args(candidate)
    model = build_model_for_hidden_dim(args, checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(args.device)

    metadata = {
        'candidate': dict(checkpoint['candidate']),
        'params_k': float(checkpoint['params_k']),
        'batch_size': int(checkpoint['batch_size']),
        'device': args.device,
    }
    return model, metadata


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-dir', type=str)
    parser.add_argument('--budget', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_cli_args()

    if cli_args.worker_dir:
        if cli_args.budget is None:
            raise RuntimeError('--budget is required in worker mode.')
        run_training_session(cli_args.worker_dir, cli_args.budget)
    else:
        run_summary = run_validation_experiment(
            build_worker_command,
            load_checkpoint_model,
        )
        summary = {
            'selected_candidate': run_summary['candidate'],
            'params_k': float(run_summary['params_k']),
            'artifact_path': run_summary['artifact_path'],
            'log_path': run_summary['log_path'],
            'final_val_ap': float(run_summary['val_ap']),
        }
        print(f"Selected candidate: {run_summary['candidate'].get('name', 'unnamed')}")
        print(f"Params k: {run_summary['params_k']:.1f}")
        print(f"Artifact path: {run_summary['artifact_path']}")
        print(f"Log path: {run_summary['log_path']}")
        print(f"Final val: {run_summary['val_ap']:.3f}")
        print(json.dumps(summary, sort_keys=True))
