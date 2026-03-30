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

8. Verify the runtime again inside that worktree before starting experiments:

```bash
python -c "import torch, torch_geometric; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch_geometric.__version__)"
```

Do not assume the parent repository environment automatically carries over to a new worktree. If the worktree uses its own virtual environment, install or activate the required GPU-capable packages there before continuing.

If the environment uses `uv`, run the same commands through `uv run`.

## Experiment Discipline

Every logged row should represent one conceptual experiment, not every inner candidate sweep.
The next experiment should usually be chosen only after inspecting the previous result.
Do not turn the campaign into a fixed 50-run hyperparameter grid unless the user explicitly asks for that.
Even when the user asks to finish all 50 experiments, still work adaptively in small batches, inspect the newest evidence, and then choose the next batch. A request to complete the full campaign is not permission to predeclare the whole schedule or blindly queue dozens of runs from one initial guess.

Within one active campaign:

1. Run the untouched baseline first.
2. Make the paper-driven change in [`train.py`](train.py) that currently seems most promising.
3. Run `python prepare.py run-and-log --short-caption "<caption>" --description "<text>"`.
4. Read the returned validation result, status, log path, and artifact path.
5. Use the paper plus the latest results to decide the next change.
6. Repeat until the campaign reaches its target experiment count, usually 50.

Stay paper-aligned:

- transfer the paper's central mechanism, normalization, routing, fusion, aggregation, or readout ideas
- prefer larger architectural or training-process changes early when they are paper-motivated
- use small paper-motivated sweeps only when they help refine a promising current idea
- avoid unrelated architecture churn
- never use test AP to choose the next experiment

Avoid defaulting to a predeclared experiment catalog:

- do not hardcode dozens of future experiment ids or configs and then execute them blindly
- do not let automation choose future experiments without first considering the newest validation evidence
- helper scripts for setup or logging are fine, but they should not replace iterative research judgment
- if a user explicitly asks for the whole campaign, prefer short adaptive batches with a brief checkpoint after each batch over one long unattended sweep
- if you do run an inner sweep inside one experiment, keep it tightly scoped and describe the shared hypothesis in the logged row

## Logging

Use the helper, never hand-edit TSV rows:

```bash
python prepare.py run-and-log --short-caption "<caption>" --description "<text>"
```

That command runs `train.py`, captures stdout/stderr to a log file, evaluates the recoverable checkpoint, and writes the ledger row without requiring ad hoc shell parsing.

Rules:

- `short_caption` must be readable and at most 22 characters
- `run-and-log` is the standard experiment entrypoint
- if training fails or times out, `run-and-log` should still write a `crash` row with `val_ap 0` and `params_k 0`
- `artifact_path` is required for every non-crash row
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

Users should normally run experiments through `prepare.py run-and-log`.
When run directly, [`train.py`](train.py) should print:

- selected candidate
- parameter count
- artifact path
- log path
- final validation AP
- a machine-readable JSON summary line at the end for tooling

## Analysis

Run the notebook after experiments have been logged:

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

The notebook plots validation progress across campaigns and summarizes the one-shot final test field from `campaigns.tsv`.

## Persistence

Do not stop just because one experiment underperforms. Keep iterating until the requested experiment count is complete or the user explicitly redirects the campaign.
