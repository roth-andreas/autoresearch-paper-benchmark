import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed benchmark and repository contract)
# ---------------------------------------------------------------------------
DATASET = 'Peptides-func'
TIME_BUDGET = 60
MAX_PARAMS = 50000
DATA_ROOT = './data/lrgb'
SPLITS = ('train', 'val', 'test')
BEST_CHECKPOINT_NAME = 'best_checkpoint.pt'

RESULTS_PATH = 'results.tsv'
CAMPAIGNS_PATH = 'campaigns.tsv'
LOCAL_STATE_DIR = 'results'
ACTIVE_CAMPAIGN_PATH = os.path.join(LOCAL_STATE_DIR, 'current_campaign.json')
SHARED_STATE_ENV = 'AUTORESEARCH_SHARED_DIR'
SHARED_STATE_DIRNAME = 'autoresearch'
ARTIFACTS_DIRNAME = 'artifacts'
LOCK_FILENAME = '.ledger.lock'
TRAIN_SNAPSHOT_NAME = 'train_snapshot.py'
CHECKPOINT_SUMMARY_NAME = 'checkpoint_summary.json'
FINAL_TEST_SUMMARY_NAME = 'final_test_summary.json'
DEFAULT_BASELINE_REF = 'main'
DEFAULT_IDEA_BLOCK_SIZE = 5

CAMPAIGN_COLUMNS = (
    'campaign_id',
    'paper_label',
    'paper_url',
    'baseline_ref',
    'start_branch',
    'target_experiments',
    'idea_block_size',
    'created_at',
    'notes',
    'final_test_experiment_num',
    'final_test_commit',
    'final_test_ap',
    'final_test_artifact_path',
    'final_test_timestamp',
)

RESULTS_COLUMNS = (
    'campaign_id',
    'paper_label',
    'experiment_num',
    'commit',
    'val_ap',
    'params_k',
    'status',
    'short_caption',
    'description',
    'block_id',
    'block_label',
    'block_run_index',
    'block_size',
    'block_role',
    'seed_experiment_num',
    'branch',
    'log_path',
    'artifact_path',
    'timestamp',
)

RESULT_STATUSES = {'keep', 'discard', 'crash'}
BLOCK_ROLES = {'seed', 'tune'}


class ExperimentRunError(RuntimeError):
    def __init__(self, message, artifact_dir='', log_path=''):
        super().__init__(message)
        self.artifact_dir = artifact_dir
        self.log_path = log_path


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_dataset(split='train'):
    return LRGBDataset(root=DATA_ROOT, name=DATASET, split=split)


def _get_stacked_attr_max(attr_name):
    max_vals = []
    for split in SPLITS:
        dataset = get_dataset(split)
        values = getattr(dataset._data, attr_name)
        max_vals.append(values.max(dim=0).values)
    return torch.stack(max_vals, dim=0).max(dim=0).values


def get_feature_dims():
    return (_get_stacked_attr_max('x') + 1).tolist()


def get_edge_feature_dims():
    return (_get_stacked_attr_max('edge_attr') + 1).tolist()


def get_num_tasks():
    return int(get_dataset('train')[0].y.size(-1))


def get_loaders(batch_size):
    train_dataset = get_dataset('train')
    val_dataset = get_dataset('val')
    test_dataset = get_dataset('test')
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_train_val_loaders(batch_size):
    train_loader, val_loader, _ = get_loaders(batch_size)
    return train_loader, val_loader


def get_val_loader(batch_size):
    _, val_loader = get_train_val_loaders(batch_size)
    return val_loader


def get_test_loader(batch_size):
    _, _, test_loader = get_loaders(batch_size)
    return test_loader


def _average_precision_score(y_true, y_pred):
    ap_list = []

    for task_idx in range(y_true.shape[1]):
        true_col = y_true[:, task_idx]
        pred_col = y_pred[:, task_idx]

        labeled_mask = ~np.isnan(true_col)
        true_col = true_col[labeled_mask]
        pred_col = pred_col[labeled_mask]

        positives = true_col == 1
        negatives = true_col == 0
        if positives.sum() == 0 or negatives.sum() == 0:
            continue

        order = np.argsort(-pred_col)
        sorted_true = true_col[order]
        tp = (sorted_true == 1).astype(np.float64)
        fp = (sorted_true == 0).astype(np.float64)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum)
        ap = float((precision * tp).sum() / tp.sum())
        ap_list.append(ap)

    if not ap_list:
        raise RuntimeError('No valid Peptides-func labels available for AP computation.')

    return float(sum(ap_list) / len(ap_list))


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

        num_graphs = batch.num_graphs
        total_loss += loss.item() * num_graphs
        total_graphs += num_graphs

        y_true.append(targets.cpu())
        y_pred.append(torch.sigmoid(logits).cpu())

    mean_loss = total_loss / max(total_graphs, 1)
    ap = _average_precision_score(
        torch.cat(y_true, dim=0).numpy(),
        torch.cat(y_pred, dim=0).numpy(),
    )
    return ap, mean_loss


@torch.no_grad()
def final_test(model, test_loader, device):
    test_ap, _ = evaluate_loader(model, test_loader, device)
    return test_ap


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _normalize_text(value):
    return ' '.join(str(value).split()).strip()


def _normalize_required_text(value, field_name):
    value = _normalize_text(value)
    if not value:
        raise ValueError(f'{field_name} must be non-empty.')
    return value


def _normalize_positive_int(value, field_name):
    normalized_value = int(value)
    if normalized_value <= 0:
        raise ValueError(f'{field_name} must be positive, got {value!r}.')
    return normalized_value


def _normalize_optional_positive_int(value, field_name):
    text = _normalize_text(value)
    if not text:
        return None
    return _normalize_positive_int(text, field_name)


def _normalize_short_caption(short_caption, max_len=22):
    value = _normalize_text(short_caption)
    if not value:
        raise ValueError('short_caption must be non-empty.')
    if len(value) > max_len:
        raise ValueError(
            f'short_caption must be at most {max_len} characters, got {len(value)}.'
        )
    return value


def _safe_float(value, default=float('nan')):
    text = _normalize_text(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_int(value, default=0):
    text = _normalize_text(value)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _normalize_block_role(block_role):
    value = _normalize_text(block_role).lower()
    if not value:
        return ''
    if value not in BLOCK_ROLES:
        raise ValueError(f'Invalid block_role {block_role!r}. Expected one of {sorted(BLOCK_ROLES)}.')
    return value


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _run_git_command(*args, cwd=None):
    completed = subprocess.run(
        ['git', *args],
        cwd=cwd or os.getcwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _safe_current_branch():
    try:
        return _run_git_command('branch', '--show-current')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def _safe_git_toplevel():
    try:
        return _run_git_command('rev-parse', '--show-toplevel')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getcwd()


def _git_ref_exists(ref_name):
    try:
        _run_git_command('rev-parse', '--verify', ref_name)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def _git_tag_exists(tag_name):
    return _git_ref_exists(f'refs/tags/{tag_name}')


def _git_branch_exists(branch_name):
    return _git_ref_exists(f'refs/heads/{branch_name}')


def _slugify(text):
    value = _normalize_text(text).lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    value = value.strip('-')
    return value or 'paper'


def _paper_token_from_url(paper_url):
    paper_url = _normalize_text(paper_url)
    arxiv_match = re.search(
        r'arxiv\\.org/(?:abs|pdf)/([0-9]{4}\\.[0-9]{4,5})(?:v[0-9]+)?',
        paper_url,
    )
    if arxiv_match:
        return arxiv_match.group(1).replace('.', '-')
    return ''


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
def get_shared_state_dir():
    override = os.environ.get(SHARED_STATE_ENV)
    if override:
        return os.path.abspath(override)
    try:
        common_dir = _run_git_command('rev-parse', '--git-common-dir')
    except (subprocess.CalledProcessError, FileNotFoundError):
        common_dir = LOCAL_STATE_DIR
    return os.path.join(os.path.abspath(common_dir), SHARED_STATE_DIRNAME)


def get_shared_results_path():
    return os.path.join(get_shared_state_dir(), RESULTS_PATH)


def get_shared_campaigns_path():
    return os.path.join(get_shared_state_dir(), CAMPAIGNS_PATH)


def get_shared_artifacts_dir():
    return os.path.join(get_shared_state_dir(), ARTIFACTS_DIRNAME)


def get_campaign_artifacts_dir(campaign_id):
    campaign_id = _normalize_required_text(campaign_id, 'campaign_id')
    return os.path.join(get_shared_artifacts_dir(), campaign_id)


def _local_active_campaign_path(root_dir=None):
    base_dir = os.path.abspath(root_dir or os.getcwd())
    return os.path.join(base_dir, ACTIVE_CAMPAIGN_PATH)


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _initialize_tsv(path, columns):
    _ensure_parent_dir(path)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)


def initialize_results_tsv(path=RESULTS_PATH):
    _initialize_tsv(path, RESULTS_COLUMNS)


def initialize_campaigns_tsv(path=CAMPAIGNS_PATH):
    _initialize_tsv(path, CAMPAIGN_COLUMNS)


def initialize_shared_ledgers():
    results_path = get_shared_results_path()
    campaigns_path = get_shared_campaigns_path()
    os.makedirs(get_shared_artifacts_dir(), exist_ok=True)
    _ensure_tsv_schema(results_path, RESULTS_COLUMNS)
    _ensure_tsv_schema(campaigns_path, CAMPAIGN_COLUMNS)
    return results_path, campaigns_path


def _load_tsv_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        return list(reader.fieldnames or []), list(reader)


def _load_result_rows(path):
    _, rows = _load_tsv_rows(path)
    return rows


def _load_campaign_rows(path):
    _, rows = _load_tsv_rows(path)
    return rows


def _rewrite_tsv(path, columns, rows):
    _ensure_parent_dir(path)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in columns})


def _ensure_tsv_schema(path, columns):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _initialize_tsv(path, columns)
        return

    fieldnames, rows = _load_tsv_rows(path)
    if tuple(fieldnames) != tuple(columns):
        _rewrite_tsv(path, columns, rows)


def _write_active_campaign_id(campaign_id, path=ACTIVE_CAMPAIGN_PATH):
    _ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'campaign_id': campaign_id}, handle, indent=2)
        handle.write('\n')


def get_active_campaign_id(path=None):
    path = path or _local_active_campaign_path()
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return _normalize_text(payload.get('campaign_id', ''))


def sync_local_snapshots(root_dir=None):
    results_path, campaigns_path = initialize_shared_ledgers()
    target_root = os.path.abspath(root_dir or os.getcwd())
    shutil.copyfile(results_path, os.path.join(target_root, RESULTS_PATH))
    shutil.copyfile(campaigns_path, os.path.join(target_root, CAMPAIGNS_PATH))


@contextmanager
def _shared_lock(timeout_seconds=60.0, poll_seconds=0.1):
    state_dir = get_shared_state_dir()
    os.makedirs(state_dir, exist_ok=True)
    lock_path = os.path.join(state_dir, LOCK_FILENAME)
    deadline = time.time() + timeout_seconds

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(f'{os.getpid()}\n')
            break
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f'Timed out waiting for shared ledger lock: {lock_path}')
            time.sleep(poll_seconds)

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Campaign registry
# ---------------------------------------------------------------------------
def _campaign_row_by_id(campaign_id, path=None):
    campaign_id = _normalize_required_text(campaign_id, 'campaign_id')
    campaigns_path = path or get_shared_campaigns_path()
    for row in _load_campaign_rows(campaigns_path):
        if _normalize_text(row.get('campaign_id', '')) == campaign_id:
            return row
    return None


def list_campaigns(path=None):
    campaigns_path = path or get_shared_campaigns_path()
    initialize_shared_ledgers()
    rows = _load_campaign_rows(campaigns_path)
    return sorted(rows, key=lambda row: row.get('created_at', ''))


def _existing_campaign_ids():
    return {
        _normalize_text(row.get('campaign_id', ''))
        for row in list_campaigns()
        if _normalize_text(row.get('campaign_id', ''))
    }


def _derive_campaign_id(paper_url='', paper_label=''):
    token = _paper_token_from_url(paper_url) or _slugify(paper_label)
    base_id = f'paper-{token}'
    existing_ids = _existing_campaign_ids()
    run_index = 1
    while True:
        candidate = f'{base_id}-r{run_index}'
        if candidate not in existing_ids:
            return candidate
        run_index += 1


def _default_worktree_path(campaign_id):
    top_level = os.path.abspath(_safe_git_toplevel())
    parent_dir = os.path.dirname(top_level)
    repo_name = os.path.basename(top_level)
    return os.path.join(parent_dir, f'{repo_name}-{campaign_id}')


def register_campaign(
    campaign_id,
    paper_label,
    paper_url='',
    baseline_ref='',
    start_branch='',
    target_experiments=50,
    idea_block_size=DEFAULT_IDEA_BLOCK_SIZE,
    notes='',
    local_root=None,
):
    campaign_id = _normalize_required_text(campaign_id, 'campaign_id')
    paper_label = _normalize_required_text(paper_label, 'paper_label')
    paper_url = _normalize_text(paper_url)
    baseline_ref = _normalize_text(baseline_ref)
    start_branch = _normalize_text(start_branch) or _safe_current_branch()
    target_experiments = _normalize_positive_int(target_experiments, 'target_experiments')
    idea_block_size = _normalize_positive_int(idea_block_size, 'idea_block_size')
    notes = _normalize_text(notes)

    with _shared_lock():
        initialize_shared_ledgers()
        existing = _campaign_row_by_id(campaign_id)
        if existing is not None:
            mismatch_fields = []
            expected = {
                'paper_label': paper_label,
                'paper_url': paper_url,
                'baseline_ref': baseline_ref,
                'start_branch': start_branch,
                'target_experiments': str(target_experiments),
                'idea_block_size': str(idea_block_size),
            }
            for field_name, expected_value in expected.items():
                existing_value = _normalize_text(existing.get(field_name, ''))
                if field_name == 'idea_block_size' and not existing_value:
                    existing_value = str(DEFAULT_IDEA_BLOCK_SIZE)
                if existing_value != _normalize_text(expected_value):
                    mismatch_fields.append(field_name)
            if mismatch_fields:
                raise ValueError(
                    f'Campaign {campaign_id!r} already exists with different metadata: '
                    f'{", ".join(mismatch_fields)}.'
                )
        else:
            with open(get_shared_campaigns_path(), 'a', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
                writer.writerow([
                    campaign_id,
                    paper_label,
                    paper_url,
                    baseline_ref,
                    start_branch,
                    str(target_experiments),
                    str(idea_block_size),
                    _utc_now_iso(),
                    notes,
                    '',
                    '',
                    '',
                    '',
                    '',
                ])

    _write_active_campaign_id(campaign_id, _local_active_campaign_path(local_root))
    sync_local_snapshots(local_root)
    return _campaign_row_by_id(campaign_id)


def use_campaign(campaign_id, local_root=None):
    initialize_shared_ledgers()
    campaign = _campaign_row_by_id(campaign_id)
    if campaign is None:
        raise ValueError(f'Unknown campaign_id {campaign_id!r}.')
    _write_active_campaign_id(campaign['campaign_id'], _local_active_campaign_path(local_root))
    sync_local_snapshots(local_root)
    return campaign


def show_active_campaign(path=None):
    campaign_id = get_active_campaign_id(path)
    if not campaign_id:
        return None
    return _campaign_row_by_id(campaign_id)


def _campaign_idea_block_size(campaign):
    if campaign is None:
        return DEFAULT_IDEA_BLOCK_SIZE
    return _safe_int(campaign.get('idea_block_size', ''), default=DEFAULT_IDEA_BLOCK_SIZE)


def _block_rows_for_campaign(campaign_id, block_id, path=None):
    results_path = path or get_shared_results_path()
    rows = [
        row for row in _load_result_rows(results_path)
        if (
            _normalize_text(row.get('campaign_id', '')) == campaign_id
            and _normalize_text(row.get('block_id', '')) == block_id
        )
    ]
    return sorted(rows, key=lambda row: _safe_int(row.get('experiment_num', ''), default=0))


def _resolve_block_metadata(
    campaign,
    experiment_num,
    results_path,
    block_id='',
    block_label='',
    block_run_index=None,
    block_size=None,
    block_role='',
    seed_experiment_num=None,
):
    normalized_block_label = _normalize_text(block_label)
    normalized_block_id = _normalize_text(block_id) or (
        _slugify(normalized_block_label) if normalized_block_label else ''
    )
    normalized_block_role = _normalize_block_role(block_role)
    normalized_block_run_index = _normalize_optional_positive_int(
        block_run_index, 'block_run_index'
    )
    normalized_block_size = _normalize_optional_positive_int(block_size, 'block_size')
    normalized_seed_experiment_num = _normalize_optional_positive_int(
        seed_experiment_num, 'seed_experiment_num'
    )

    any_block_metadata = any(
        [
            normalized_block_id,
            normalized_block_label,
            normalized_block_run_index is not None,
            normalized_block_size is not None,
            normalized_block_role,
            normalized_seed_experiment_num is not None,
        ]
    )
    if not any_block_metadata:
        return {
            'block_id': '',
            'block_label': '',
            'block_run_index': '',
            'block_size': '',
            'block_role': '',
            'seed_experiment_num': '',
        }

    if not normalized_block_id:
        raise ValueError(
            'Block logging requires --block-id or --block-label so the experiment '
            'can be assigned to an idea block.'
        )

    existing_rows = _block_rows_for_campaign(
        campaign_id=campaign['campaign_id'],
        block_id=normalized_block_id,
        path=results_path,
    )
    campaign_block_size = _campaign_idea_block_size(campaign)

    if existing_rows:
        existing_label = _normalize_text(existing_rows[0].get('block_label', ''))
        existing_block_size = _safe_int(
            existing_rows[0].get('block_size', ''),
            default=campaign_block_size,
        )
        existing_seed_experiment_num = _safe_int(
            existing_rows[0].get('seed_experiment_num', ''),
            default=_safe_int(existing_rows[0].get('experiment_num', ''), default=0),
        )
        next_block_run_index = max(
            _safe_int(row.get('block_run_index', ''), default=0) for row in existing_rows
        ) + 1

        if normalized_block_label and existing_label and normalized_block_label != existing_label:
            raise ValueError(
                f'block_label {normalized_block_label!r} does not match the existing '
                f'label {existing_label!r} for block_id {normalized_block_id!r}.'
            )
        normalized_block_label = normalized_block_label or existing_label

        if normalized_block_size is None:
            normalized_block_size = existing_block_size
        elif normalized_block_size != existing_block_size:
            raise ValueError(
                f'block_size {normalized_block_size} does not match the existing '
                f'block size {existing_block_size} for block_id {normalized_block_id!r}.'
            )

        if normalized_block_run_index is None:
            normalized_block_run_index = next_block_run_index
        elif normalized_block_run_index != next_block_run_index:
            raise ValueError(
                f'Next run in block {normalized_block_id!r} must use block_run_index '
                f'{next_block_run_index}, got {normalized_block_run_index}.'
            )

        if normalized_seed_experiment_num is None:
            normalized_seed_experiment_num = existing_seed_experiment_num
        elif normalized_seed_experiment_num != existing_seed_experiment_num:
            raise ValueError(
                f'seed_experiment_num {normalized_seed_experiment_num} does not match '
                f'the existing seed experiment {existing_seed_experiment_num} for block '
                f'{normalized_block_id!r}.'
            )
    else:
        if not normalized_block_label:
            raise ValueError(
                'The first run in an idea block requires --block-label so the larger '
                'paper idea is recorded in the ledger.'
            )
        if normalized_block_size is None:
            normalized_block_size = campaign_block_size
        if normalized_block_run_index is None:
            normalized_block_run_index = 1
        if normalized_block_run_index != 1:
            raise ValueError(
                f'The first run in block {normalized_block_id!r} must use block_run_index 1.'
            )
        if normalized_seed_experiment_num is None:
            normalized_seed_experiment_num = int(experiment_num)
        elif normalized_seed_experiment_num != int(experiment_num):
            raise ValueError(
                'The first run in a block must use its own experiment number as '
                'seed_experiment_num.'
            )

    if normalized_block_run_index > normalized_block_size:
        raise ValueError(
            f'Block {normalized_block_id!r} has size {normalized_block_size}, so it '
            f'cannot accept block_run_index {normalized_block_run_index}.'
        )

    if not normalized_block_role:
        normalized_block_role = 'seed' if normalized_block_run_index == 1 else 'tune'

    if normalized_block_run_index == 1 and normalized_block_role != 'seed':
        raise ValueError('The first run in an idea block must use block_role "seed".')
    if normalized_block_run_index > 1 and normalized_block_role != 'tune':
        raise ValueError('Follow-up runs in an idea block must use block_role "tune".')

    return {
        'block_id': normalized_block_id,
        'block_label': normalized_block_label,
        'block_run_index': str(normalized_block_run_index),
        'block_size': str(normalized_block_size),
        'block_role': normalized_block_role,
        'seed_experiment_num': str(normalized_seed_experiment_num),
    }


def list_campaign_blocks(campaign_id=''):
    campaign_id = _normalize_text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected. Use use-campaign first.')

    initialize_shared_ledgers()
    rows = [
        row for row in _load_result_rows(get_shared_results_path())
        if (
            _normalize_text(row.get('campaign_id', '')) == campaign_id
            and _normalize_text(row.get('block_id', ''))
        )
    ]

    grouped = {}
    for row in rows:
        block_id = _normalize_text(row.get('block_id', ''))
        grouped.setdefault(block_id, []).append(row)

    summaries = []
    for block_id, block_rows in grouped.items():
        block_rows.sort(key=lambda row: _safe_int(row.get('experiment_num', ''), default=0))
        best_row = max(
            block_rows,
            key=lambda row: _safe_float(row.get('val_ap', ''), default=float('-inf')),
        )
        block_size = _safe_int(
            block_rows[0].get('block_size', ''),
            default=DEFAULT_IDEA_BLOCK_SIZE,
        )
        summaries.append(
            {
                'block_id': block_id,
                'block_label': _normalize_text(block_rows[0].get('block_label', '')),
                'runs_logged': len(block_rows),
                'block_size': block_size,
                'completed': len(block_rows) >= block_size,
                'seed_experiment_num': _safe_int(
                    block_rows[0].get('seed_experiment_num', ''),
                    default=_safe_int(block_rows[0].get('experiment_num', ''), default=0),
                ),
                'best_experiment_num': _safe_int(best_row.get('experiment_num', ''), default=0),
                'best_val_ap': _safe_float(best_row.get('val_ap', ''), default=float('nan')),
                'first_experiment_num': _safe_int(
                    block_rows[0].get('experiment_num', ''),
                    default=0,
                ),
                'last_experiment_num': _safe_int(
                    block_rows[-1].get('experiment_num', ''),
                    default=0,
                ),
            }
        )

    return sorted(summaries, key=lambda row: row['first_experiment_num'])


def _bootstrap_summary(
    campaign_id,
    paper_label,
    paper_url,
    baseline_ref,
    start_branch,
    idea_block_size,
    worktree_path,
    baseline_exists,
    baseline_created,
    worktree_created,
    requires_create_baseline_from=False,
):
    return {
        'campaign_id': campaign_id,
        'paper_label': paper_label,
        'paper_url': paper_url,
        'baseline_ref': baseline_ref,
        'start_branch': start_branch,
        'idea_block_size': int(idea_block_size),
        'worktree_path': os.path.abspath(worktree_path),
        'baseline_exists': baseline_exists,
        'baseline_created': baseline_created,
        'worktree_created': worktree_created,
        'requires_create_baseline_from': bool(requires_create_baseline_from),
    }


def bootstrap_campaign(
    paper_label,
    paper_url,
    campaign_id='',
    baseline_ref='',
    branch='',
    worktree_path='',
    target_experiments=50,
    idea_block_size=DEFAULT_IDEA_BLOCK_SIZE,
    notes='',
    create_baseline_from='',
    dry_run=False,
):
    initialize_shared_ledgers()

    paper_label = _normalize_required_text(paper_label, 'paper_label')
    paper_url = _normalize_text(paper_url)
    campaign_id = _normalize_text(campaign_id) or _derive_campaign_id(
        paper_url=paper_url,
        paper_label=paper_label,
    )
    baseline_ref = _normalize_text(baseline_ref) or DEFAULT_BASELINE_REF
    branch = _normalize_text(branch) or f'paper/{campaign_id}'
    worktree_path = _normalize_text(worktree_path) or _default_worktree_path(campaign_id)
    target_experiments = _normalize_positive_int(target_experiments, 'target_experiments')
    idea_block_size = _normalize_positive_int(idea_block_size, 'idea_block_size')
    notes = _normalize_text(notes)
    create_baseline_from = _normalize_text(create_baseline_from)

    baseline_exists = _git_tag_exists(baseline_ref) or _git_ref_exists(baseline_ref)
    branch_exists = _git_branch_exists(branch)
    worktree_exists = os.path.exists(worktree_path)
    baseline_created = False
    worktree_created = False

    if not baseline_exists and not create_baseline_from:
        if dry_run:
            return _bootstrap_summary(
                campaign_id=campaign_id,
                paper_label=paper_label,
                paper_url=paper_url,
                baseline_ref=baseline_ref,
                start_branch=branch,
                idea_block_size=idea_block_size,
                worktree_path=worktree_path,
                baseline_exists=False,
                baseline_created=False,
                worktree_created=not worktree_exists,
                requires_create_baseline_from=True,
            )
        raise ValueError(
            f'Baseline ref {baseline_ref!r} does not exist. '
            'Pass --create-baseline-from <git-ref> to create it explicitly.'
        )

    if worktree_exists and not branch_exists:
        raise ValueError(
            f'Worktree path already exists: {worktree_path!r}. '
            'Choose a different --worktree-path or reuse the existing branch.'
        )

    if dry_run:
        return _bootstrap_summary(
            campaign_id=campaign_id,
            paper_label=paper_label,
            paper_url=paper_url,
            baseline_ref=baseline_ref,
            start_branch=branch,
            idea_block_size=idea_block_size,
            worktree_path=worktree_path,
            baseline_exists=baseline_exists,
            baseline_created=not baseline_exists,
            worktree_created=not worktree_exists,
        )

    if not baseline_exists:
        _run_git_command('tag', baseline_ref, create_baseline_from)
        baseline_created = True

    if not worktree_exists:
        if branch_exists:
            _run_git_command('worktree', 'add', worktree_path, branch)
        else:
            _run_git_command('worktree', 'add', worktree_path, '-b', branch, baseline_ref)
        worktree_created = True

    register_campaign(
        campaign_id=campaign_id,
        paper_label=paper_label,
        paper_url=paper_url,
        baseline_ref=baseline_ref,
        start_branch=branch,
        target_experiments=target_experiments,
        idea_block_size=idea_block_size,
        notes=notes,
        local_root=worktree_path,
    )
    sync_local_snapshots()
    sync_local_snapshots(worktree_path)
    return _bootstrap_summary(
        campaign_id=campaign_id,
        paper_label=paper_label,
        paper_url=paper_url,
        baseline_ref=baseline_ref,
        start_branch=branch,
        idea_block_size=idea_block_size,
        worktree_path=worktree_path,
        baseline_exists=True,
        baseline_created=baseline_created,
        worktree_created=worktree_created,
    )


# ---------------------------------------------------------------------------
# Experiment artifacts and logging
# ---------------------------------------------------------------------------
def create_experiment_artifact_dir(campaign_id):
    artifact_root = get_campaign_artifacts_dir(campaign_id)
    os.makedirs(artifact_root, exist_ok=True)
    return tempfile.mkdtemp(prefix='exp-', dir=artifact_root)


def _snapshot_train_source(artifact_dir):
    shutil.copyfile(
        os.path.join(os.getcwd(), 'train.py'),
        os.path.join(artifact_dir, TRAIN_SNAPSHOT_NAME),
    )


def _write_json(path, payload):
    _ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def _next_experiment_num(campaign_id, path=None):
    results_path = path or get_shared_results_path()
    max_experiment_num = 0
    for row in _load_result_rows(results_path):
        if _normalize_text(row.get('campaign_id', '')) != campaign_id:
            continue
        experiment_text = _normalize_text(row.get('experiment_num', ''))
        if experiment_text:
            max_experiment_num = max(max_experiment_num, int(experiment_text))
    return max_experiment_num + 1


def _result_row_by_experiment_num(campaign_id, experiment_num, path=None):
    results_path = path or get_shared_results_path()
    for row in _load_result_rows(results_path):
        if _normalize_text(row.get('campaign_id', '')) != campaign_id:
            continue
        if _safe_int(row.get('experiment_num', ''), default=0) == int(experiment_num):
            return row
    return None


def _append_result_dict(row, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        initialize_results_tsv(path)
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_COLUMNS,
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writerow(row)


def append_result_row(
    commit,
    val_ap,
    params_k,
    status,
    short_caption,
    description,
    block_id='',
    block_label='',
    block_run_index=None,
    block_size=None,
    block_role='',
    seed_experiment_num=None,
    campaign_id='',
    branch='',
    log_path='',
    artifact_path='',
    experiment_num=None,
    path=None,
):
    normalized_status = _normalize_required_text(status, 'status').lower()
    if normalized_status not in RESULT_STATUSES:
        raise ValueError(f'Invalid status {status!r}. Expected one of {sorted(RESULT_STATUSES)}.')

    campaign_id = _normalize_text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError(
            'No active campaign selected. Use "python prepare.py start-campaign ..." '
            'or "python prepare.py use-campaign --campaign-id ..." first.'
        )

    campaign = _campaign_row_by_id(campaign_id)
    if campaign is None:
        raise ValueError(f'Unknown campaign_id {campaign_id!r}.')

    branch = _normalize_text(branch) or _safe_current_branch() or _normalize_text(
        campaign.get('start_branch', '')
    )
    description = _normalize_text(description)
    short_caption = _normalize_short_caption(short_caption)
    log_path = _normalize_text(log_path)
    artifact_path = _normalize_text(artifact_path)
    if normalized_status != 'crash' and not artifact_path:
        raise ValueError('artifact_path is required for non-crash experiment rows.')
    results_path = path or get_shared_results_path()

    with _shared_lock():
        initialize_shared_ledgers()
        if experiment_num is None:
            experiment_num = _next_experiment_num(campaign['campaign_id'], results_path)
        experiment_num = _normalize_positive_int(experiment_num, 'experiment_num')

        for row in _load_result_rows(results_path):
            same_campaign = _normalize_text(row.get('campaign_id', '')) == campaign['campaign_id']
            same_experiment = _normalize_text(row.get('experiment_num', '')) == str(experiment_num)
            if same_campaign and same_experiment:
                raise ValueError(
                    f'Experiment {experiment_num} already exists for campaign '
                    f'{campaign["campaign_id"]!r}.'
                )

        block_metadata = _resolve_block_metadata(
            campaign=campaign,
            experiment_num=experiment_num,
            results_path=results_path,
            block_id=block_id,
            block_label=block_label,
            block_run_index=block_run_index,
            block_size=block_size,
            block_role=block_role,
            seed_experiment_num=seed_experiment_num,
        )

        row = {
            'campaign_id': campaign['campaign_id'],
            'paper_label': _normalize_text(campaign.get('paper_label', '')),
            'experiment_num': str(experiment_num),
            'commit': _normalize_required_text(commit, 'commit'),
            'val_ap': f'{float(val_ap):.6f}',
            'params_k': f'{float(params_k):.1f}',
            'status': normalized_status,
            'short_caption': short_caption,
            'description': description,
            'block_id': block_metadata['block_id'],
            'block_label': block_metadata['block_label'],
            'block_run_index': block_metadata['block_run_index'],
            'block_size': block_metadata['block_size'],
            'block_role': block_metadata['block_role'],
            'seed_experiment_num': block_metadata['seed_experiment_num'],
            'branch': branch,
            'log_path': log_path,
            'artifact_path': artifact_path,
            'timestamp': _utc_now_iso(),
        }
        _append_result_dict(row, results_path)

    sync_local_snapshots()
    return experiment_num


def _terminate_process(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_validation_experiment(
    build_worker_command_fn,
    load_checkpoint_fn,
    campaign_id='',
    log_path='',
):
    campaign_id = _normalize_text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected for validation-only evaluation.')

    artifact_dir = create_experiment_artifact_dir(campaign_id)
    _snapshot_train_source(artifact_dir)

    checkpoint_path = os.path.join(artifact_dir, BEST_CHECKPOINT_NAME)
    command = build_worker_command_fn(artifact_dir, TIME_BUDGET)
    resolved_log_path = _normalize_text(log_path)
    if not resolved_log_path:
        resolved_log_path = os.path.join(artifact_dir, 'worker.log')

    process = None
    timed_out = False
    log_handle = None

    try:
        if resolved_log_path:
            _ensure_parent_dir(resolved_log_path)
            log_handle = open(resolved_log_path, 'w', encoding='utf-8')
            process = subprocess.Popen(
                command,
                cwd=os.getcwd(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            process = subprocess.Popen(command, cwd=os.getcwd())

        try:
            process.wait(timeout=TIME_BUDGET)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(process)
    finally:
        if log_handle is not None:
            log_handle.close()

    if timed_out:
        raise ExperimentRunError(
            'Training worker timed out before producing a recoverable checkpoint.',
            artifact_dir=artifact_dir,
            log_path=resolved_log_path,
        )

    if process.returncode != 0:
        raise ExperimentRunError(
            f'Training worker failed with exit code {process.returncode}.',
            artifact_dir=artifact_dir,
            log_path=resolved_log_path,
        )

    if not os.path.exists(checkpoint_path):
        raise ExperimentRunError(
            'Training ended without writing a recoverable best checkpoint.',
            artifact_dir=artifact_dir,
            log_path=resolved_log_path,
        )

    model, metadata = load_checkpoint_fn(checkpoint_path)
    val_loader = get_val_loader(metadata['batch_size'])
    val_ap, _ = evaluate_loader(model, val_loader, metadata['device'])

    _write_json(
        os.path.join(artifact_dir, CHECKPOINT_SUMMARY_NAME),
        {
            'campaign_id': campaign_id,
            'candidate': metadata.get('candidate', {}),
            'params_k': float(metadata['params_k']),
            'batch_size': int(metadata['batch_size']),
            'val_ap': float(val_ap),
            'artifact_path': artifact_dir,
            'checkpoint_path': checkpoint_path,
            'log_path': resolved_log_path,
            'timestamp': _utc_now_iso(),
        },
    )

    return {
        'campaign_id': campaign_id,
        'val_ap': float(val_ap),
        'params_k': float(metadata['params_k']),
        'candidate': dict(metadata.get('candidate', {})),
        'batch_size': int(metadata['batch_size']),
        'artifact_path': artifact_dir,
        'checkpoint_path': checkpoint_path,
        'log_path': resolved_log_path,
    }


def _load_train_module_from_path(train_path, module_name):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Missing train source: {train_path}')

    previous_prepare = sys.modules.get('prepare')
    sys.modules['prepare'] = sys.modules[__name__]
    try:
        spec = importlib.util.spec_from_file_location(module_name, train_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_prepare is not None:
            sys.modules['prepare'] = previous_prepare
        else:
            sys.modules.pop('prepare', None)

    return module


def run_and_log_experiment(
    short_caption,
    description,
    status='auto',
    block_id='',
    block_label='',
    block_run_index=None,
    block_size=None,
    block_role='',
    seed_experiment_num=None,
    commit='',
    campaign_id='',
    branch='',
    log_path='',
    experiment_num=None,
):
    campaign_id = _normalize_text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected. Use use-campaign first.')

    resolved_commit = _normalize_text(commit)
    if not resolved_commit:
        resolved_commit = _run_git_command('rev-parse', '--short', 'HEAD')

    try:
        train_module = _load_train_module_from_path(
            os.path.join(os.getcwd(), 'train.py'),
            'train_current_run_and_log',
        )
        run_summary = run_validation_experiment(
            train_module.build_worker_command,
            train_module.load_checkpoint_model,
            campaign_id=campaign_id,
            log_path=log_path,
        )
    except Exception as error:
        crash_log_path = error.log_path if isinstance(error, ExperimentRunError) else ''
        experiment_num = append_result_row(
            commit=resolved_commit,
            val_ap=0.0,
            params_k=0.0,
            status='crash',
            short_caption=short_caption,
            description=description,
            block_id=block_id,
            block_label=block_label,
            block_run_index=block_run_index,
            block_size=block_size,
            block_role=block_role,
            seed_experiment_num=seed_experiment_num,
            campaign_id=campaign_id,
            branch=branch,
            log_path=crash_log_path,
            artifact_path='',
            experiment_num=experiment_num,
        )
        result_row = _result_row_by_experiment_num(campaign_id, experiment_num)
        return {
            'campaign_id': campaign_id,
            'commit': resolved_commit,
            'status': 'crash',
            'experiment_num': int(experiment_num),
            'short_caption': _normalize_short_caption(short_caption),
            'description': _normalize_text(description),
            'val_ap': 0.0,
            'params_k': 0.0,
            'artifact_path': '',
            'log_path': crash_log_path,
            'error': str(error),
            'block_id': _normalize_text((result_row or {}).get('block_id', '')),
            'block_label': _normalize_text((result_row or {}).get('block_label', '')),
            'block_run_index': _normalize_text((result_row or {}).get('block_run_index', '')),
            'block_size': _normalize_text((result_row or {}).get('block_size', '')),
            'block_role': _normalize_text((result_row or {}).get('block_role', '')),
            'seed_experiment_num': _normalize_text(
                (result_row or {}).get('seed_experiment_num', '')
            ),
        }

    normalized_status = _normalize_text(status).lower() or 'auto'
    if normalized_status == 'auto':
        prior_best = float('-inf')
        for row in _load_result_rows(get_shared_results_path()):
            if _normalize_text(row.get('campaign_id', '')) != campaign_id:
                continue
            prior_best = max(prior_best, _safe_float(row.get('val_ap', ''), default=float('-inf')))
        normalized_status = 'keep' if run_summary['val_ap'] >= prior_best else 'discard'

    experiment_num = append_result_row(
        commit=resolved_commit,
        val_ap=run_summary['val_ap'],
        params_k=run_summary['params_k'],
        status=normalized_status,
        short_caption=short_caption,
        description=description,
        block_id=block_id,
        block_label=block_label,
        block_run_index=block_run_index,
        block_size=block_size,
        block_role=block_role,
        seed_experiment_num=seed_experiment_num,
        campaign_id=campaign_id,
        branch=branch,
        log_path=run_summary['log_path'],
        artifact_path=run_summary['artifact_path'],
        experiment_num=experiment_num,
    )
    result_row = _result_row_by_experiment_num(campaign_id, experiment_num)

    run_summary.update(
        {
            'commit': resolved_commit,
            'status': normalized_status,
            'experiment_num': int(experiment_num),
            'short_caption': _normalize_short_caption(short_caption),
            'description': _normalize_text(description),
            'block_id': _normalize_text((result_row or {}).get('block_id', '')),
            'block_label': _normalize_text((result_row or {}).get('block_label', '')),
            'block_run_index': _normalize_text((result_row or {}).get('block_run_index', '')),
            'block_size': _normalize_text((result_row or {}).get('block_size', '')),
            'block_role': _normalize_text((result_row or {}).get('block_role', '')),
            'seed_experiment_num': _normalize_text(
                (result_row or {}).get('seed_experiment_num', '')
            ),
        }
    )
    return run_summary


def _load_train_snapshot_module(artifact_path):
    train_path = os.path.join(artifact_path, TRAIN_SNAPSHOT_NAME)
    module_name = f'train_snapshot_{os.path.basename(artifact_path)}'
    return _load_train_module_from_path(train_path, module_name)


def finalize_campaign_test(campaign_id='', experiment_num=None, force=False):
    campaign_id = _normalize_text(campaign_id) or get_active_campaign_id()
    if not campaign_id:
        raise ValueError('No active campaign selected. Use use-campaign first.')

    with _shared_lock():
        initialize_shared_ledgers()
        campaign = _campaign_row_by_id(campaign_id)
        if campaign is None:
            raise ValueError(f'Unknown campaign_id {campaign_id!r}.')

        if _normalize_text(campaign.get('final_test_ap', '')) and not force:
            raise ValueError(
                f'Campaign {campaign_id!r} already has a final test result. '
                'Pass --force to recompute it.'
            )

        rows = [
            row for row in _load_result_rows(get_shared_results_path())
            if _normalize_text(row.get('campaign_id', '')) == campaign_id
        ]
        if not rows:
            raise ValueError(f'Campaign {campaign_id!r} has no logged experiments.')

        target_experiments = int(_normalize_text(campaign.get('target_experiments', '0')) or 0)
        max_experiment = max(int(_normalize_text(row.get('experiment_num', '0')) or 0) for row in rows)
        if (not force) and target_experiments and max_experiment < target_experiments:
            raise ValueError(
                f'Campaign {campaign_id!r} has only {max_experiment} experiments logged; '
                f'expected {target_experiments} before final test.'
            )

        recoverable_rows = [
            row for row in rows if _normalize_text(row.get('artifact_path', ''))
        ]
        if not recoverable_rows:
            raise ValueError(
                f'Campaign {campaign_id!r} has no recoverable experiment artifacts.'
            )

        if experiment_num is None:
            selected_row = max(
                recoverable_rows,
                key=lambda row: (
                    _safe_float(row.get('val_ap', ''), default=float('-inf')),
                    -int(_normalize_text(row.get('experiment_num', '0')) or 0),
                ),
            )
        else:
            selected_matches = [
                row for row in recoverable_rows
                if int(_normalize_text(row.get('experiment_num', '0')) or 0) == int(experiment_num)
            ]
            if not selected_matches:
                raise ValueError(
                    f'Experiment {experiment_num} not found with a recoverable artifact '
                    f'for campaign {campaign_id!r}.'
                )
            selected_row = selected_matches[0]

    artifact_path = _normalize_required_text(selected_row.get('artifact_path', ''), 'artifact_path')
    checkpoint_path = os.path.join(artifact_path, BEST_CHECKPOINT_NAME)
    train_module = _load_train_snapshot_module(artifact_path)
    model, metadata = train_module.load_checkpoint_model(checkpoint_path)
    test_loader = get_test_loader(metadata['batch_size'])
    test_ap = final_test(model, test_loader, metadata['device'])

    summary = {
        'campaign_id': campaign_id,
        'selected_experiment_num': int(selected_row['experiment_num']),
        'selected_commit': _normalize_text(selected_row.get('commit', '')),
        'val_ap': float(selected_row['val_ap']),
        'test_ap': float(test_ap),
        'artifact_path': artifact_path,
        'timestamp': _utc_now_iso(),
    }
    _write_json(os.path.join(artifact_path, FINAL_TEST_SUMMARY_NAME), summary)

    with _shared_lock():
        campaigns_path = get_shared_campaigns_path()
        _, campaign_rows = _load_tsv_rows(campaigns_path)
        for row in campaign_rows:
            if _normalize_text(row.get('campaign_id', '')) != campaign_id:
                continue
            row['final_test_experiment_num'] = str(summary['selected_experiment_num'])
            row['final_test_commit'] = summary['selected_commit']
            row['final_test_ap'] = f"{summary['test_ap']:.6f}"
            row['final_test_artifact_path'] = artifact_path
            row['final_test_timestamp'] = summary['timestamp']
            break
        _rewrite_tsv(campaigns_path, CAMPAIGN_COLUMNS, campaign_rows)

    sync_local_snapshots()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description=(
            'Benchmark harness and campaign ledger for autonomous paper-driven '
            'Peptides-func experiments.'
        )
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    subparsers.add_parser(
        'init-results',
        help='Initialize the shared results and campaign ledgers.',
    )

    bootstrap_parser = subparsers.add_parser(
        'bootstrap-campaign',
        help='Create or reuse a paper campaign worktree from a shared baseline.',
    )
    bootstrap_parser.add_argument('--paper-label', required=True)
    bootstrap_parser.add_argument('--paper-url', default='')
    bootstrap_parser.add_argument('--campaign-id', default='')
    bootstrap_parser.add_argument('--baseline-ref', default='')
    bootstrap_parser.add_argument('--branch', default='')
    bootstrap_parser.add_argument('--worktree-path', default='')
    bootstrap_parser.add_argument('--target-experiments', type=int, default=50)
    bootstrap_parser.add_argument('--idea-block-size', type=int, default=DEFAULT_IDEA_BLOCK_SIZE)
    bootstrap_parser.add_argument('--notes', default='')
    bootstrap_parser.add_argument('--create-baseline-from', default='')
    bootstrap_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the derived campaign plan without creating anything.',
    )

    start_parser = subparsers.add_parser(
        'start-campaign',
        help='Register an already-created campaign in the shared ledger.',
    )
    start_parser.add_argument('--campaign-id', required=True)
    start_parser.add_argument('--paper-label', required=True)
    start_parser.add_argument('--paper-url', default='')
    start_parser.add_argument('--baseline-ref', default='')
    start_parser.add_argument('--start-branch', default='')
    start_parser.add_argument('--target-experiments', type=int, default=50)
    start_parser.add_argument('--idea-block-size', type=int, default=DEFAULT_IDEA_BLOCK_SIZE)
    start_parser.add_argument('--notes', default='')

    use_parser = subparsers.add_parser(
        'use-campaign',
        help='Mark an existing campaign as active in this worktree.',
    )
    use_parser.add_argument('--campaign-id', required=True)

    subparsers.add_parser('show-campaign', help='Show the active local campaign.')
    subparsers.add_parser('list-campaigns', help='List all registered campaigns.')
    show_blocks_parser = subparsers.add_parser(
        'show-blocks',
        help='Summarize logged idea blocks for the active or selected campaign.',
    )
    show_blocks_parser.add_argument('--campaign-id', default='')
    subparsers.add_parser('show-paths', help='Print shared ledger and snapshot paths.')

    finalize_parser = subparsers.add_parser(
        'finalize-campaign-test',
        help='Run the one-shot test evaluation for a completed campaign.',
    )
    finalize_parser.add_argument('--campaign-id', default='')
    finalize_parser.add_argument('--experiment-num', type=int)
    finalize_parser.add_argument('--force', action='store_true')

    log_parser = subparsers.add_parser(
        'run-and-log',
        help='Run train.py, capture a log, and append the experiment result.',
    )
    log_parser.add_argument('--campaign-id', default='')
    log_parser.add_argument('--commit', default='')
    log_parser.add_argument(
        '--status',
        default='auto',
        choices=['auto', 'keep', 'discard'],
    )
    log_parser.add_argument('--short-caption', required=True)
    log_parser.add_argument('--description', required=True)
    log_parser.add_argument('--block-id', default='')
    log_parser.add_argument('--block-label', default='')
    log_parser.add_argument('--block-run-index', type=int)
    log_parser.add_argument('--block-size', type=int)
    log_parser.add_argument(
        '--block-role',
        default='',
        choices=['', 'seed', 'tune'],
        help='Optional role inside the idea block. Defaults to seed for run 1 and tune otherwise.',
    )
    log_parser.add_argument('--seed-experiment-num', type=int)
    log_parser.add_argument('--branch', default='')
    log_parser.add_argument(
        '--log-path',
        default='',
        help='Optional explicit log file path. Defaults to <artifact_dir>/worker.log.',
    )
    log_parser.add_argument('--experiment-num', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    cli_args = _parse_cli_args()

    if cli_args.command == 'init-results':
        initialize_shared_ledgers()
        sync_local_snapshots()
    elif cli_args.command == 'bootstrap-campaign':
        summary = bootstrap_campaign(
            paper_label=cli_args.paper_label,
            paper_url=cli_args.paper_url,
            campaign_id=cli_args.campaign_id,
            baseline_ref=cli_args.baseline_ref,
            branch=cli_args.branch,
            worktree_path=cli_args.worktree_path,
            target_experiments=cli_args.target_experiments,
            idea_block_size=cli_args.idea_block_size,
            notes=cli_args.notes,
            create_baseline_from=cli_args.create_baseline_from,
            dry_run=cli_args.dry_run,
        )
        print(json.dumps(summary, indent=2))
    elif cli_args.command == 'start-campaign':
        register_campaign(
            campaign_id=cli_args.campaign_id,
            paper_label=cli_args.paper_label,
            paper_url=cli_args.paper_url,
            baseline_ref=cli_args.baseline_ref,
            start_branch=cli_args.start_branch,
            target_experiments=cli_args.target_experiments,
            idea_block_size=cli_args.idea_block_size,
            notes=cli_args.notes,
        )
    elif cli_args.command == 'use-campaign':
        use_campaign(cli_args.campaign_id)
    elif cli_args.command == 'show-campaign':
        active_campaign = show_active_campaign()
        if active_campaign is None:
            print('No active campaign selected.')
        else:
            print(json.dumps(active_campaign, indent=2))
    elif cli_args.command == 'list-campaigns':
        campaigns = list_campaigns()
        if not campaigns:
            print('No campaigns registered.')
        else:
            print('\\t'.join(CAMPAIGN_COLUMNS))
            for campaign in campaigns:
                print('\\t'.join(str(campaign.get(column, '')) for column in CAMPAIGN_COLUMNS))
    elif cli_args.command == 'show-blocks':
        blocks = list_campaign_blocks(cli_args.campaign_id)
        if not blocks:
            print('No idea blocks logged.')
        else:
            columns = (
                'block_id',
                'block_label',
                'runs_logged',
                'block_size',
                'completed',
                'seed_experiment_num',
                'best_experiment_num',
                'best_val_ap',
                'first_experiment_num',
                'last_experiment_num',
            )
            print('\\t'.join(columns))
            for block in blocks:
                print('\\t'.join(str(block.get(column, '')) for column in columns))
    elif cli_args.command == 'show-paths':
        print(f'shared_state_dir\\t{get_shared_state_dir()}')
        print(f'shared_results_tsv\\t{get_shared_results_path()}')
        print(f'shared_campaigns_tsv\\t{get_shared_campaigns_path()}')
        print(f'shared_artifacts_dir\\t{get_shared_artifacts_dir()}')
        print(f'local_results_tsv\\t{os.path.abspath(RESULTS_PATH)}')
        print(f'local_campaigns_tsv\\t{os.path.abspath(CAMPAIGNS_PATH)}')
        print(f'active_campaign_file\\t{os.path.abspath(ACTIVE_CAMPAIGN_PATH)}')
    elif cli_args.command == 'finalize-campaign-test':
        summary = finalize_campaign_test(
            campaign_id=cli_args.campaign_id,
            experiment_num=cli_args.experiment_num,
            force=cli_args.force,
        )
        print(json.dumps(summary, indent=2))
    elif cli_args.command == 'run-and-log':
        summary = run_and_log_experiment(
            short_caption=cli_args.short_caption,
            description=cli_args.description,
            status=cli_args.status,
            block_id=cli_args.block_id,
            block_label=cli_args.block_label,
            block_run_index=cli_args.block_run_index,
            block_size=cli_args.block_size,
            block_role=cli_args.block_role,
            seed_experiment_num=cli_args.seed_experiment_num,
            commit=cli_args.commit,
            campaign_id=cli_args.campaign_id,
            branch=cli_args.branch,
            log_path=cli_args.log_path,
            experiment_num=cli_args.experiment_num,
        )
        print(json.dumps(summary, indent=2))
