# autoresearch

This repository runs autonomous paper-to-benchmark transfer experiments on one fixed graph benchmark.

## Core Contract

- [`prepare.py`](prepare.py) defines the benchmark and campaign infrastructure.
- [`train.py`](train.py) is the editable model and training code.
- Normal experiments should only modify [`train.py`](train.py).

The benchmark is fixed:

- dataset: `Peptides-func`
- split: `train / val / test`
- per-experiment metric: validation AP
- wall-clock budget: `TIME_BUDGET`
- parameter cap: `MAX_PARAMS`

The test set is not for experiment selection. It is used exactly once per campaign, after the target number of experiments has been completed.

## Campaign Model

Each paper gets its own campaign.

Recommended pattern:

1. Keep one neutral baseline ref, for example `neutral-baseline`.
2. Start every paper campaign from that same baseline ref.
3. Use one git worktree per campaign so multiple agents can run in parallel.
4. Register or bootstrap the campaign with [`prepare.py`](prepare.py).
5. Log each experiment to the shared ledger.
6. Finalize the test once the campaign reaches its target count.

Do not start a new paper from a previous paper's best commit.

## Setup

For a new paper campaign:

1. Read [`README.md`](README.md), [`prepare.py`](prepare.py), and [`train.py`](train.py).
2. Read the paper carefully enough to identify the transferable ideas.
3. Initialize the shared ledgers if needed:

```bash
python prepare.py init-results
```

If PyTorch cannot see CUDA on the local machine, fix the environment first by installing a PyTorch build that matches the local driver/CUDA setup in the current environment. Do not hardcode a single CUDA wheel in repo config for one machine.

4. Bootstrap a new campaign:

```bash
python prepare.py bootstrap-campaign --paper-label "<paper title>" --paper-url <paper url>
```

5. If the baseline ref does not exist yet, bootstrap with:

```bash
python prepare.py bootstrap-campaign --paper-label "<paper title>" --paper-url <paper url> --create-baseline-from <neutral-baseline-ref>
```

6. Continue work inside the returned `worktree_path`.
7. Confirm the active campaign:

```bash
python prepare.py show-campaign
```

If the environment uses `uv`, run the same commands through `uv run`.

## Experiment Discipline

Every logged row should represent one conceptual experiment, not every inner candidate sweep.

Within one active campaign:

1. Run the untouched baseline first.
2. Make one paper-driven change in [`train.py`](train.py).
3. Run `python train.py`.
4. Capture the reported validation result and artifact path.
5. Log one row with `prepare.py log-result`.
6. Continue from the strongest validation idea.
7. Repeat until the campaign reaches its target experiment count, usually 50.

Stay paper-aligned:

- transfer the paper's central mechanism, normalization, routing, fusion, aggregation, or readout ideas
- test small paper-motivated sweeps when needed
- avoid unrelated architecture churn
- never use test AP to choose the next experiment

## Logging

Use the helper, never hand-edit TSV rows:

```bash
python prepare.py log-result --commit <hash> --val-ap <ap> --params-k <k> --status <keep|discard|crash> --short-caption "<caption>" --description "<text>" --log-path <logfile> --artifact-path <artifact-dir>
```

Rules:

- `short_caption` must be readable and at most 22 characters
- `artifact_path` is required for every non-crash row
- crashes should still be logged with `val_ap 0` and `params_k 0`
- the active campaign is taken from `results/current_campaign.json` unless `--campaign-id` is supplied

`results.tsv` columns:

```text
campaign_id	paper_label	experiment_num	commit	val_ap	params_k	status	short_caption	description	branch	log_path	artifact_path	timestamp
```

## Final Test

After the campaign reaches the target experiment count:

```bash
python prepare.py finalize-campaign-test
```

That command selects the recoverable experiment with the highest validation AP and runs the test set once. The resulting test score is stored in `campaigns.tsv`, not in `results.tsv`.

`campaigns.tsv` columns:

```text
campaign_id	paper_label	paper_url	baseline_ref	start_branch	target_experiments	created_at	notes	final_test_experiment_num	final_test_commit	final_test_ap	final_test_artifact_path	final_test_timestamp
```

## Runtime Contract

[`prepare.py`](prepare.py) owns:

- dataset loading
- metric computation
- hard experiment timeout
- shared ledgers
- checkpoint artifact directory management
- final one-shot test evaluation

[`train.py`](train.py) owns:

- architecture
- optimizer and schedule
- regularization
- checkpointing the best validation model during training
- optional inner candidate sweeps inside one experiment

When run normally, [`train.py`](train.py) should print:

- selected candidate
- parameter count
- artifact path
- final validation AP

## Analysis

Run the notebook after experiments have been logged:

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

The notebook plots validation progress across campaigns and summarizes the one-shot final test field from `campaigns.tsv`.

## Persistence

Do not stop just because one experiment underperforms. Keep iterating until the requested experiment count is complete or the user explicitly redirects the campaign.
