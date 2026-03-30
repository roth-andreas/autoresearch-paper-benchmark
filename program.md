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

1. Keep one shared baseline ref, for example `main`.
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
python prepare.py bootstrap-campaign --paper-label "<paper title>" --paper-url <paper url> --create-baseline-from <main-ref>
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

Every logged row should represent one experiment execution, but a 50-run paper campaign should usually be organized into idea blocks rather than a flat sequence of tiny tweaks.
The default campaign shape is one untouched baseline plus paper-aligned idea blocks of size 5.
Each idea block should focus on one larger paper-derived mechanism and then tune that mechanism locally.
The next larger idea should usually be chosen only after inspecting the results of the current block.
Do not turn the campaign into a fixed 50-run hyperparameter grid unless the user explicitly asks for that.
If the user's prompt says to run the full campaign, treat that as permission to continue until the target experiment count is reached unless the user explicitly interrupts or redirects the run.

Within one active campaign:

1. Run the untouched baseline first.
2. Choose one larger paper-driven idea that is meaningfully different from previous blocks.
3. Implement the seed version of that idea in [`train.py`](train.py).
4. Log the seed run with `block_run_index 1` and `block_role seed`.
5. Run up to four local follow-ups inside the same block, usually tuning layer count, width, learning rate, dropout, weight decay, batch size, or closely related settings for that same mechanism.
6. After the block finishes, inspect the block outcome and choose the next larger paper idea.
7. Repeat until the campaign reaches its target experiment count, usually 50.

Default block pattern:

- run 1 of the block: seed implementation of one larger idea
- runs 2-5 of the block: local tuning around that same idea
- block size defaults to 5 unless the campaign was created with a different `idea_block_size`
- if a seed idea is clearly non-competitive, it is acceptable to stop that block early and move to a new larger idea rather than forcing all five runs

Stay paper-aligned:

- transfer the paper's central mechanism, normalization, routing, fusion, aggregation, or readout ideas
- between blocks, prefer larger architectural or training-process changes that map to distinct paper ideas
- within a block, use small paper-motivated sweeps only to refine the current larger idea
- avoid unrelated architecture churn
- never use test AP to choose the next experiment

Avoid defaulting to a predeclared experiment catalog:

- do not hardcode all 50 future experiments and then execute them blindly
- do not let automation choose future idea blocks without first considering the newest validation evidence
- helper scripts for setup or logging are fine, but they should not replace iterative research judgment
- if a user explicitly asks for the whole campaign, prefer block-by-block progress over one long unattended sweep
- do not spend multiple blocks on tiny hyperparameter nudges unless one larger idea has already proved clearly dominant

## Logging

Use the helper, never hand-edit TSV rows:

```bash
python prepare.py run-and-log \
  --block-id "<idea-block-id>" \
  --block-label "<larger paper idea>" \
  --block-run-index 1 \
  --block-role seed \
  --short-caption "<caption>" \
  --description "<text>"
```

That command runs `train.py`, captures stdout/stderr to a log file, evaluates the recoverable checkpoint, and writes the ledger row without requiring ad hoc shell parsing.

Rules:

- `short_caption` must be readable and at most 22 characters
- `run-and-log` is the standard experiment entrypoint
- `block_id` groups all runs belonging to the same larger paper idea
- `block_label` should describe the larger idea in readable language
- `block_run_index` should usually count from `1` to the campaign's `idea_block_size`, normally `5`
- `block_role` should be `seed` for the first run in a block and `tune` for later runs
- `seed_experiment_num` is auto-filled for the first run in a block and inherited by later runs when omitted
- `show-blocks` summarizes which idea blocks have been logged so far
- if training fails or times out, `run-and-log` should still write a `crash` row with `val_ap 0` and `params_k 0`
- `artifact_path` is required for every non-crash row
- the active campaign is taken from `results/current_campaign.json` unless `--campaign-id` is supplied

Example follow-up run inside the same idea block:

```bash
python prepare.py run-and-log \
  --block-id "<idea-block-id>" \
  --block-run-index 2 \
  --block-role tune \
  --short-caption "<caption>" \
  --description "<text>"
```

`results.tsv` columns:

```text
campaign_id	paper_label	experiment_num	commit	val_ap	params_k	status	short_caption	description	block_id	block_label	block_run_index	block_size	block_role	seed_experiment_num	branch	log_path	artifact_path	timestamp
```

## Final Test

After the campaign reaches the target experiment count:

```bash
python prepare.py finalize-campaign-test
```

That command selects the recoverable experiment with the highest validation AP and runs the test set once. The resulting test score is stored in `campaigns.tsv`, not in `results.tsv`.

`campaigns.tsv` columns:

```text
campaign_id	paper_label	paper_url	baseline_ref	start_branch	target_experiments	idea_block_size	created_at	notes	final_test_experiment_num	final_test_commit	final_test_ap	final_test_artifact_path	final_test_timestamp
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
