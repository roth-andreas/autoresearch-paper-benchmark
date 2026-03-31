# autoresearch

`autoresearch` is a small repository for running paper-driven experiments on one fixed graph benchmark.

The idea is simple:

- [`prepare.py`](prepare.py) owns the benchmark, experiment logging, and final test evaluation
- [`train.py`](train.py) is where you implement model ideas from a paper
- each paper gets its own local campaign

## What Stays Fixed

Every experiment in this repository uses the same outer setup:

- dataset: `Peptides-func`
- split: `train / val / test`
- validation metric: average precision (`val_ap`)
- time budget per experiment: `300` seconds
- parameter cap: `500000`

During a campaign, you choose models using validation performance only.
The test split is used once at the end of the campaign.

## Files You Should Care About

- [`prepare.py`](prepare.py): benchmark harness and experiment runner
- [`train.py`](train.py): editable model and training code
- [`program.md`](program.md): a short workflow guide for an AI coding agent

Generated local files:

- `campaigns.tsv`
- `results.tsv`
- `results/current_campaign.json`
- `artifacts/`
- `data/`

## Setup

Install the project dependencies:

```bash
uv sync
```

Install PyTorch for your machine using the official installer:

- https://pytorch.org/get-started/locally/

Then install PyTorch Geometric into the same environment:

```bash
uv pip install torch_geometric
```

Check that your environment works:

```bash
uv run python -c "import torch, torch_geometric; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch_geometric.__version__)"
```

If you are not using `uv`, that is fine. Just make sure `python prepare.py ...` and `python train.py` both run in the same environment.

## First Campaign

Initialize the local files once:

```bash
uv run python prepare.py init-results
```

Create a campaign for a paper:

```bash
uv run python prepare.py bootstrap-campaign \
  --paper-label "What Can We Learn From MIMO Graph Convolutions?" \
  --paper-url https://arxiv.org/abs/2505.11346 \
  --target-experiments 15
```

That command creates the campaign and marks it as active.

You can check the active campaign at any time:

```bash
uv run python prepare.py show-campaign
```

## Running Experiments

The usual loop is:

1. Read the paper.
2. Implement one idea in [`train.py`](train.py).
3. Run one experiment with [`prepare.py`](prepare.py).
4. Look at the validation result.
5. Adjust [`train.py`](train.py) and repeat.

Run one experiment like this:

```bash
uv run python prepare.py run-and-log \
  --short-caption "baseline" \
  --description "Untouched baseline model"
```

What `run-and-log` does:

- creates a new artifact directory
- snapshots the current `train.py`
- runs training with the fixed time budget
- reloads the best checkpoint
- evaluates validation AP
- appends one row to `results.tsv`

You should normally edit only [`train.py`](train.py) between runs.

## Final Test

After the campaign has reached its target number of experiments, run:

```bash
uv run python prepare.py finalize-campaign-test
```

This selects the recoverable experiment with the best validation AP and evaluates the test split once.
The final test result is stored in `campaigns.tsv`.

## Result Files

`results.tsv` stores one row per experiment:

```text
campaign_id	experiment_num	commit	val_ap	params_k	status	short_caption	description	artifact_path	log_path	timestamp
```

`campaigns.tsv` stores one row per paper campaign:

```text
campaign_id	paper_label	paper_url	target_experiments	created_at	final_test_experiment_num	final_test_ap	final_test_artifact_path	final_test_timestamp
```

## Using An AI Agent

If you want an AI coding agent to run a campaign, point it to [`program.md`](program.md) and keep the prompt high-level.

Example:

```text
Please read program.md and run a full campaign for this paper.
The model ideas should follow: https://arxiv.org/abs/2505.11346
```

## Notes

- This repository is intentionally small and local-first.
- It is designed for sequential experiments, not parallel campaign management.
- If an experiment crashes, `run-and-log` still records that run in `results.tsv`.

## License

MIT
