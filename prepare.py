import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader


DATASET = 'Peptides-func'
TIME_BUDGET = 300
MAX_PARAMS = 500000
DATA_ROOT = './data/lrgb'
SPLITS = ('train', 'val', 'test')
BEST_CHECKPOINT_NAME = 'best_checkpoint.pt'

RESULTS_PATH = Path('results.tsv')
CAMPAIGNS_PATH = Path('campaigns.tsv')
STATE_DIR = Path('results')
ACTIVE_CAMPAIGN_PATH = STATE_DIR / 'current_campaign.json'
ARTIFACTS_DIR = Path('artifacts')
TRAIN_SNAPSHOT_NAME = 'train_snapshot.py'
DEFAULT_TARGET_EXPERIMENTS = 15

CAMPAIGN_COLUMNS = (
    'campaign_id',
    'paper_label',
    'paper_url',
    'target_experiments',
    'created_at',
    'final_test_experiment_num',
    'final_test_ap',
    'final_test_artifact_path',
    'final_test_timestamp',
)

RESULTS_COLUMNS = (
    'campaign_id',
    'experiment_num',
    'commit',
    'val_ap',
    'params_k',
    'status',
    'short_caption',
    'description',
    'artifact_path',
    'log_path',
    'timestamp',
)

VALID_STATUSES = {'keep', 'discard', 'crash'}


class ExperimentRunError(RuntimeError):
    def __init__(self, message, artifact_dir='', log_path=''):
        super().__init__(message)
        self.artifact_dir = artifact_dir
        self.log_path = log_path


def text(value):
    return ' '.join(str(value).split()).strip()


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ''


def slugify(value):
    slug = re.sub(r'[^a-z0-9]+', '-', text(value).lower()).strip('-')
    return slug or 'campaign'


def derive_campaign_id(paper_label='', paper_url=''):
    match = re.search(r'(\d{4}\.\d{4,5})', text(paper_url))
    base = match.group(1).replace('.', '-') if match else slugify(paper_label)
    existing = {row['campaign_id'] for row in campaign_rows() if row.get('campaign_id')}
    candidate = base
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f'{base}-r{suffix}'
    return candidate


def read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def read_rows(path, columns):
    path = Path(path)
    if not path.exists():
        write_rows(path, columns, [])
        return []

    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        rows = list(reader)
        header = list(reader.fieldnames or [])

    if header != list(columns):
        rows = [{column: text(row.get(column, '')) for column in columns} for row in rows]
        write_rows(path, columns, rows)
    return rows


def init_storage():
    STATE_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    read_rows(CAMPAIGNS_PATH, CAMPAIGN_COLUMNS)
    read_rows(RESULTS_PATH, RESULTS_COLUMNS)


def campaign_rows():
    init_storage()
    return read_rows(CAMPAIGNS_PATH, CAMPAIGN_COLUMNS)


def result_rows():
    init_storage()
    return read_rows(RESULTS_PATH, RESULTS_COLUMNS)


def save_campaign_rows(rows):
    write_rows(CAMPAIGNS_PATH, CAMPAIGN_COLUMNS, rows)


def save_result_rows(rows):
    write_rows(RESULTS_PATH, RESULTS_COLUMNS, rows)


def get_active_campaign_id():
    return text(read_json(ACTIVE_CAMPAIGN_PATH, {}).get('campaign_id', ''))


def set_active_campaign(campaign_id):
    write_json(ACTIVE_CAMPAIGN_PATH, {'campaign_id': campaign_id})


def get_campaign(campaign_id):
    for row in campaign_rows():
        if row['campaign_id'] == campaign_id:
            return row
    return None


def show_active_campaign():
    campaign_id = get_active_campaign_id()
    return get_campaign(campaign_id) if campaign_id else None


def bootstrap_campaign(paper_label, paper_url='', campaign_id='', target_experiments=DEFAULT_TARGET_EXPERIMENTS):
    paper_label = text(paper_label)
    if not paper_label:
        raise ValueError('paper_label must be non-empty.')
    target_experiments = int(target_experiments)
    if target_experiments <= 0:
        raise ValueError('target_experiments must be positive.')

    campaign_id = text(campaign_id) or derive_campaign_id(paper_label, paper_url)
    rows = campaign_rows()
    campaign = get_campaign(campaign_id)
    if campaign is None:
        campaign = {
            'campaign_id': campaign_id,
            'paper_label': paper_label,
            'paper_url': text(paper_url),
            'target_experiments': str(target_experiments),
            'created_at': now_iso(),
            'final_test_experiment_num': '',
            'final_test_ap': '',
            'final_test_artifact_path': '',
            'final_test_timestamp': '',
        }
        rows.append(campaign)
        save_campaign_rows(rows)
    set_active_campaign(campaign_id)
    return campaign


def use_campaign(campaign_id):
    campaign = get_campaign(text(campaign_id))
    if campaign is None:
        raise ValueError(f'Unknown campaign_id {campaign_id!r}.')
    set_active_campaign(campaign['campaign_id'])
    return campaign


def next_experiment_num(campaign_id):
    nums = [
        int(row['experiment_num'])
        for row in result_rows()
        if row['campaign_id'] == campaign_id and text(row.get('experiment_num', ''))
    ]
    return max(nums, default=0) + 1


def create_artifact_dir(campaign_id, experiment_num):
    path = ARTIFACTS_DIR / campaign_id / f'exp-{int(experiment_num):04d}'
    if path.exists():
        raise ValueError(f'Artifact directory already exists: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def append_result(row):
    rows = result_rows()
    rows.append({column: text(row.get(column, '')) for column in RESULTS_COLUMNS})
    save_result_rows(rows)
    return rows[-1]


@lru_cache(maxsize=None)
def get_dataset(split='train'):
    return LRGBDataset(root=DATA_ROOT, name=DATASET, split=split)


def _stacked_attr_max(attr_name):
    maxima = []
    for split in SPLITS:
        maxima.append(getattr(get_dataset(split)._data, attr_name).max(dim=0).values)
    return torch.stack(maxima, dim=0).max(dim=0).values


def get_feature_dims():
    return (_stacked_attr_max('x') + 1).tolist()


def get_edge_feature_dims():
    return (_stacked_attr_max('edge_attr') + 1).tolist()


def get_num_tasks():
    return int(get_dataset('train')[0].y.size(-1))


def get_loaders(batch_size):
    pin_memory = torch.cuda.is_available()
    datasets = [get_dataset(split) for split in SPLITS]
    return tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=index == 0,
            pin_memory=pin_memory,
        )
        for index, dataset in enumerate(datasets)
    )


def get_train_val_loaders(batch_size):
    train_loader, val_loader, _ = get_loaders(batch_size)
    return train_loader, val_loader


def average_precision_score(y_true, y_pred):
    ap_values = []
    for task_idx in range(y_true.shape[1]):
        true_col = y_true[:, task_idx]
        pred_col = y_pred[:, task_idx]
        mask = ~np.isnan(true_col)
        true_col = true_col[mask]
        pred_col = pred_col[mask]
        if (true_col == 1).sum() == 0 or (true_col == 0).sum() == 0:
            continue
        order = np.argsort(-pred_col)
        sorted_true = true_col[order]
        tp = (sorted_true == 1).astype(np.float64)
        fp = (sorted_true == 0).astype(np.float64)
        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        ap_values.append(float((precision * tp).sum() / tp.sum()))
    if not ap_values:
        raise RuntimeError('No valid Peptides-func labels available for AP computation.')
    return float(sum(ap_values) / len(ap_values))


@torch.no_grad()
def evaluate_loader(model, loader, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y.view(logits.size(0), -1).float()
        loss = criterion(logits, targets)
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        y_true.append(targets.cpu())
        y_pred.append(torch.sigmoid(logits).cpu())

    mean_loss = total_loss / max(total_graphs, 1)
    ap = average_precision_score(
        torch.cat(y_true, dim=0).numpy(),
        torch.cat(y_pred, dim=0).numpy(),
    )
    return ap, mean_loss


def terminate_process(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def load_train_module(train_path, module_name):
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f'Missing train source: {train_path}')

    previous_prepare = sys.modules.get('prepare')
    sys.modules['prepare'] = sys.modules[__name__]
    try:
        spec = importlib.util.spec_from_file_location(module_name, train_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_prepare is not None:
            sys.modules['prepare'] = previous_prepare
        else:
            sys.modules.pop('prepare', None)
    return module


def run_validation_experiment(build_worker_command_fn, load_checkpoint_fn, campaign_id='', log_path='', experiment_num=None):
    campaign_id = text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected.')

    experiment_num = experiment_num or next_experiment_num(campaign_id)
    artifact_dir = create_artifact_dir(campaign_id, experiment_num)
    shutil.copyfile('train.py', artifact_dir / TRAIN_SNAPSHOT_NAME)

    checkpoint_path = artifact_dir / BEST_CHECKPOINT_NAME
    log_path = Path(text(log_path) or artifact_dir / 'worker.log')
    command = build_worker_command_fn(str(artifact_dir), TIME_BUDGET)

    timed_out = False
    with log_path.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            process.wait(timeout=TIME_BUDGET)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process)

    if timed_out and not checkpoint_path.exists():
        raise ExperimentRunError('Training timed out before writing a checkpoint.', str(artifact_dir), str(log_path))
    if process.returncode != 0 and not (timed_out and checkpoint_path.exists()):
        raise ExperimentRunError(f'Training failed with exit code {process.returncode}.', str(artifact_dir), str(log_path))
    if not checkpoint_path.exists():
        raise ExperimentRunError('Training ended without a checkpoint.', str(artifact_dir), str(log_path))

    model, metadata = load_checkpoint_fn(checkpoint_path)
    _, val_loader, _ = get_loaders(metadata['batch_size'])
    val_ap, _ = evaluate_loader(model, val_loader, metadata['device'])
    return {
        'campaign_id': campaign_id,
        'experiment_num': int(experiment_num),
        'val_ap': float(val_ap),
        'params_k': float(metadata['params_k']),
        'candidate': dict(metadata.get('candidate', {})),
        'artifact_path': str(artifact_dir),
        'log_path': str(log_path),
    }


def run_and_log_experiment(short_caption, description, status='auto', campaign_id='', commit='', log_path='', experiment_num=None):
    campaign_id = text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected.')

    experiment_num = experiment_num or next_experiment_num(campaign_id)
    commit = text(commit) or git_commit()

    try:
        train_module = load_train_module('train.py', 'train_current_run_and_log')
        summary = run_validation_experiment(
            train_module.build_worker_command,
            train_module.load_checkpoint_model,
            campaign_id=campaign_id,
            log_path=log_path,
            experiment_num=experiment_num,
        )
    except Exception as error:
        row = append_result(
            {
                'campaign_id': campaign_id,
                'experiment_num': experiment_num,
                'commit': commit,
                'val_ap': 0.0,
                'params_k': 0.0,
                'status': 'crash',
                'short_caption': text(short_caption)[:22],
                'description': text(description),
                'artifact_path': '',
                'log_path': text(getattr(error, 'log_path', '')),
                'timestamp': now_iso(),
            }
        )
        return {'error': str(error), **row}

    status = text(status).lower() or 'auto'
    if status == 'auto':
        best_so_far = max(
            [
                float(row['val_ap'])
                for row in result_rows()
                if row['campaign_id'] == campaign_id and text(row.get('val_ap', ''))
            ] or [float('-inf')]
        )
        status = 'keep' if summary['val_ap'] >= best_so_far else 'discard'
    if status not in VALID_STATUSES - {'crash'}:
        raise ValueError(f'Invalid status {status!r}.')

    row = append_result(
        {
            'campaign_id': campaign_id,
            'experiment_num': experiment_num,
            'commit': commit,
            'val_ap': f"{summary['val_ap']:.6f}",
            'params_k': f"{summary['params_k']:.1f}",
            'status': status,
            'short_caption': text(short_caption)[:22],
            'description': text(description),
            'artifact_path': summary['artifact_path'],
            'log_path': summary['log_path'],
            'timestamp': now_iso(),
        }
    )
    return {**summary, **row}


def finalize_campaign_test(campaign_id='', experiment_num=None, force=False):
    campaign_id = text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected.')

    campaign = get_campaign(campaign_id)
    if campaign is None:
        raise ValueError(f'Unknown campaign_id {campaign_id!r}.')
    if campaign['final_test_ap'] and not force:
        raise ValueError('Final test already exists. Pass --force to recompute it.')

    rows = [row for row in result_rows() if row['campaign_id'] == campaign_id]
    if not rows:
        raise ValueError('No experiments logged for this campaign.')
    if len(rows) < int(campaign['target_experiments']) and not force:
        raise ValueError('Campaign has not reached target_experiments yet.')

    recoverable = [row for row in rows if row['artifact_path']]
    if not recoverable:
        raise ValueError('No recoverable experiment artifacts found.')

    if experiment_num is None:
        selected = max(recoverable, key=lambda row: (float(row['val_ap']), -int(row['experiment_num'])))
    else:
        selected = next(
            (row for row in recoverable if int(row['experiment_num']) == int(experiment_num)),
            None,
        )
        if selected is None:
            raise ValueError(f'Experiment {experiment_num} does not have a recoverable artifact.')

    artifact_path = Path(selected['artifact_path'])
    checkpoint_path = artifact_path / BEST_CHECKPOINT_NAME
    train_module = load_train_module(artifact_path / TRAIN_SNAPSHOT_NAME, f"train_snapshot_{artifact_path.name}")
    model, metadata = train_module.load_checkpoint_model(checkpoint_path)
    _, _, test_loader = get_loaders(metadata['batch_size'])
    test_ap, _ = evaluate_loader(model, test_loader, metadata['device'])

    campaigns = campaign_rows()
    for row in campaigns:
        if row['campaign_id'] != campaign_id:
            continue
        row['final_test_experiment_num'] = selected['experiment_num']
        row['final_test_ap'] = f'{test_ap:.6f}'
        row['final_test_artifact_path'] = str(artifact_path)
        row['final_test_timestamp'] = now_iso()
        summary = row.copy()
        break
    save_campaign_rows(campaigns)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description='Sequential benchmark harness for Peptides-func experiments.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('init-results')

    bootstrap = subparsers.add_parser('bootstrap-campaign')
    bootstrap.add_argument('--paper-label', required=True)
    bootstrap.add_argument('--paper-url', default='')
    bootstrap.add_argument('--campaign-id', default='')
    bootstrap.add_argument('--target-experiments', type=int, default=DEFAULT_TARGET_EXPERIMENTS)

    use = subparsers.add_parser('use-campaign')
    use.add_argument('--campaign-id', required=True)

    subparsers.add_parser('show-campaign')
    subparsers.add_parser('list-campaigns')

    log = subparsers.add_parser('run-and-log')
    log.add_argument('--campaign-id', default='')
    log.add_argument('--commit', default='')
    log.add_argument('--status', default='auto', choices=['auto', 'keep', 'discard'])
    log.add_argument('--short-caption', required=True)
    log.add_argument('--description', required=True)
    log.add_argument('--log-path', default='')
    log.add_argument('--experiment-num', type=int)

    finalize = subparsers.add_parser('finalize-campaign-test')
    finalize.add_argument('--campaign-id', default='')
    finalize.add_argument('--experiment-num', type=int)
    finalize.add_argument('--force', action='store_true')

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    if args.command == 'init-results':
        init_storage()
    elif args.command == 'bootstrap-campaign':
        print(json.dumps(
            bootstrap_campaign(
                paper_label=args.paper_label,
                paper_url=args.paper_url,
                campaign_id=args.campaign_id,
                target_experiments=args.target_experiments,
            ),
            indent=2,
        ))
    elif args.command == 'use-campaign':
        print(json.dumps(use_campaign(args.campaign_id), indent=2))
    elif args.command == 'show-campaign':
        campaign = show_active_campaign()
        print('No active campaign selected.' if campaign is None else json.dumps(campaign, indent=2))
    elif args.command == 'list-campaigns':
        campaigns = campaign_rows()
        if not campaigns:
            print('No campaigns registered.')
        else:
            print('\t'.join(CAMPAIGN_COLUMNS))
            for row in campaigns:
                print('\t'.join(row.get(column, '') for column in CAMPAIGN_COLUMNS))
    elif args.command == 'run-and-log':
        print(json.dumps(
            run_and_log_experiment(
                short_caption=args.short_caption,
                description=args.description,
                status=args.status,
                campaign_id=args.campaign_id,
                commit=args.commit,
                log_path=args.log_path,
                experiment_num=args.experiment_num,
            ),
            indent=2,
        ))
    elif args.command == 'finalize-campaign-test':
        print(json.dumps(
            finalize_campaign_test(
                campaign_id=args.campaign_id,
                experiment_num=args.experiment_num,
                force=args.force,
            ),
            indent=2,
        ))
