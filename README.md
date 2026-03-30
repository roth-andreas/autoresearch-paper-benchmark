# autoresearch

`autoresearch` is a small framework for autonomous, paper-driven graph benchmark research. The benchmark contract lives in [`prepare.py`](prepare.py), paper ideas are implemented in [`train.py`](train.py), and the experiment loop is described in [`program.md`](program.md).

The repository is designed around one rule: each paper campaign should be comparable under the same outer harness. That means fixed data, fixed split, fixed wall-clock budget, fixed parameter cap, validation-only experiment logging, and exactly one final test evaluation after the campaign is complete.

## What Is Fixed

- dataset: `Peptides-func`
- split: `train / val / test`
- per-experiment metric: validation AP
- time budget: `TIME_BUDGET` seconds
- model cap: `MAX_PARAMS` parameters
- final holdout test: run once after the campaign reaches its target experiment count

During a campaign, every experiment saves a recoverable checkpoint artifact and logs only `val_ap`. After the target number of experiments, `prepare.py finalize-campaign-test` reloads the best-validation artifact and evaluates the test split once.

## Repository Layout

- [`prepare.py`](prepare.py): fixed benchmark harness, campaign registry, shared ledger helpers, validation evaluation, and final one-shot test.
- [`train.py`](train.py): editable model and training loop.
- [`program.md`](program.md): autonomous agent instructions.
- [`analysis.ipynb`](analysis.ipynb): notebook for validation-progress plots and campaign summaries.

Generated local files are ignored by git:

- `data/`
- `results.tsv`
- `campaigns.tsv`
- `results/current_campaign.json`
- `progress.png`
- `run-*.log`

## Quick Start

Install the project environment and general tooling with `uv`:

```bash
uv sync
```

Install PyTorch for your local machine using the official PyTorch installer:

- latest selector: https://pytorch.org/get-started/locally/
- previous versions: https://pytorch.org/get-started/previous-versions/

Choose the build that matches your operating system, Python version, package manager, and CUDA/driver situation. The repository intentionally does not hardcode a single CUDA build, because that is not portable across different NVIDIA setups.

If you are using `uv`, take the `pip install ...` command from the PyTorch site and run the equivalent `uv pip install ...` command so the package lands in the same environment as this repository.

Then install PyTorch Geometric:

```bash
uv pip install torch_geometric
```

For the basic models in this repository, a minimal PyG install is enough. If you later need the optional accelerated PyG extension packages, install the wheel set that matches your installed PyTorch and CUDA versions by following the official PyG installation guide:

- https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html

Verify the runtime before starting experiments:

```bash
uv run python -c "import torch, torch_geometric; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch_geometric.__version__)"
```

If `torch.cuda.is_available()` returns `False`, fix the local PyTorch install first before running experiments.

If the default project environment does not see CUDA, do not continue with CPU-only runs.
Instead, create or activate any environment that satisfies all of the following:

- same Python compatibility as the repo (`>=3.10`)
- PyTorch installed from the official selector or previous-versions page with a build that matches the local OS, driver, and CUDA/runtime situation
- `torch_geometric` installed into that same environment
- repository dependencies from [`pyproject.toml`](pyproject.toml) installed into that same environment

The environment manager does not matter. `uv`, `conda`, `venv`, and similar tools are all acceptable as long as this repository's Python commands run in the same GPU-capable environment.

Before starting a campaign, verify the exact environment you plan to use:

```bash
python -c "import torch, torch_geometric; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch_geometric.__version__)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required for experiments on this machine'"
```

If that `python` command is not the project environment you intend to use, prefix repository commands with the activation or runner mechanism of your environment manager, for example `uv run`, `conda run -n <env>`, or an activated shell.

Once PyTorch and PyG are working, initialize the shared ledgers:

```bash
uv run python prepare.py init-results
```

Bootstrap a new paper campaign from `main`:

```bash
uv run python prepare.py bootstrap-campaign \
  --paper-label "MIMO Graph Conv" \
  --paper-url https://arxiv.org/abs/2505.11346 \
  --target-experiments 50
```

By default, `bootstrap-campaign` starts from `main`. If you want to bootstrap from a different trusted ref, pass `--baseline-ref <git-ref>`.

If `main` is not available as a baseline ref in your local repository state, create the desired baseline explicitly from a trusted branch or commit:

```bash
uv run python prepare.py bootstrap-campaign \
  --paper-label "MIMO Graph Conv" \
  --paper-url https://arxiv.org/abs/2505.11346 \
  --create-baseline-from main \
  --target-experiments 50
```

The command prints a summary including:

- `campaign_id`
- `start_branch`
- `baseline_ref`
- `worktree_path`

Open that worktree and continue there.
Verify the runtime again inside that worktree before running experiments, because a fresh worktree may create or use a different local environment than the parent repository.

## Using An AI Agent

[`program.md`](program.md) is meant to be a lightweight agent skill. Open your coding agent in this repository, point it at `program.md`, and ask it to start with setup before it begins running experiments.

A good starter prompt is:

```text
Hi have a look at program.md and let's kick off a new experiment! Do the setup first, then run this campaign all the way through 50 experiments iteratively.
Run the campaign in 5-experiment idea blocks: each block should test one larger paper idea and then spend the rest of the block on local tuning around that idea.
The models should follow the ideas of this paper: https://arxiv.org/abs/2505.11346
```

If your agent supports permission modes, restricted mode is a good default while you are getting comfortable with the workflow.

The intended research style is adaptive, but the default unit of adaptation should be the idea block, not the single tiny tweak. For a 50-run campaign, the preferred pattern is one untouched baseline followed by paper-aligned blocks of about 5 runs each: one seed implementation of a larger paper idea plus up to four local tuning runs around that same mechanism. The default should not be to hardcode dozens of future experiment configs or auto-run the full campaign as one large hyperparameter grid.

The expected flow is:

1. The agent reads `README.md`, `prepare.py`, `train.py`, and `program.md`.
2. The agent bootstraps or activates a paper campaign.
3. The agent runs the baseline first.
4. The agent iterates on `train.py` in paper-aligned idea blocks, usually 5 runs per larger idea.
5. The agent logs one validation result per experiment, including idea-block metadata.
6. After the campaign reaches its target count, the agent runs the one-shot final test.

## Multiple Agents

To explore several papers at once, run one agent per campaign worktree.

Recommended pattern:

1. Keep one neutral baseline ref.
2. Create one campaign worktree per paper with `prepare.py bootstrap-campaign`.
3. Open one Codex, Claude Code, or similar session in each worktree.
4. Give each agent the same style of prompt, but with a different paper URL.
5. Let every agent log into the shared ledger.
6. Compare the campaigns later in `analysis.ipynb`.

Example prompts for two parallel papers:

```text
Hi have a look at program.md and let's kick off a new experiment! Do the setup first, then run this campaign all the way through 50 experiments iteratively.
Run the campaign in 5-experiment idea blocks: each block should test one larger paper idea and then spend the rest of the block on local tuning around that idea.
The models should follow the ideas of this paper: https://arxiv.org/abs/2505.11346
```

```text
Hi have a look at program.md and let's kick off a new experiment! Do the setup first, then run this campaign all the way through 50 experiments iteratively.
Run the campaign in 5-experiment idea blocks: each block should test one larger paper idea and then spend the rest of the block on local tuning around that idea.
The models should follow the ideas of this paper: https://arxiv.org/abs/2105.14491
```

The important part is that each agent works in its own campaign worktree, while all worktrees share the same ledger and artifact store.

## Why Worktrees

The recommended way to run multiple papers in parallel is one git repository with one `git worktree` per campaign.

This is cleaner than making multiple full copies because:

- every paper starts from the same neutral baseline
- each campaign gets isolated code changes and branch history
- all worktrees share one git common dir
- the authoritative ledgers live once under `<git-common-dir>/autoresearch/`
- concurrent logging is protected by a file lock
- there is no manual result merge step

You can override the shared state location with `AUTORESEARCH_SHARED_DIR`, but the default shared git-common-dir layout is usually the right choice.

## Running Experiments

Inside a campaign worktree:

1. Run the untouched baseline first.
2. Choose one larger paper-driven idea and implement the seed version in [`train.py`](train.py).
3. Log that seed with `run-and-log` using a new `block_id` and `block_label`.
4. Run up to four nearby tuning experiments within the same block.
5. Decide the next larger paper idea only after reading the current block outcome, then repeat until the campaign reaches its target count.
6. Finalize the holdout test once.

The campaign should mostly look like iterative research, not a blind search:

- early blocks should be willing to make larger architectural or training-process changes when the paper suggests them
- later blocks can narrow in with smaller follow-up changes once a strong direction appears
- small paper-motivated sweeps are allowed within a block, but they should support one local hypothesis rather than replace the overall idea-level loop
- do not precommit to a full 50-experiment schedule up front unless the user explicitly asks for that style
- the default block size is `5`, controlled by the campaign's `idea_block_size`

Example experiment command:

```bash
uv run python prepare.py run-and-log \
  --block-id lmgc-readout \
  --block-label "LMGC with richer readout" \
  --block-run-index 1 \
  --block-role seed \
  --short-caption "LMGC readout" \
  --description "Seed run for the LMGC richer-readout idea block"
```

Rules:

- `val_ap` is the only per-experiment score stored in `results.tsv`
- `prepare.py run-and-log` is the standard way to execute and record an experiment
- `block_id` groups the larger paper idea, while `block_run_index` records the run position inside that idea block
- `show-blocks` summarizes completed and in-progress idea blocks for the active campaign
- `artifact_path` is required for every non-crash row
- crashes should still be logged with status `crash`, `val_ap 0`, and `params_k 0`; `run-and-log` handles that automatically
- the test set must not be used during the campaign to choose which experiment to keep

## Final Test Once

After the campaign reaches its target experiment count, run:

```bash
uv run python prepare.py finalize-campaign-test
```

That command:

- reads the active campaign
- selects the recoverable experiment row with the highest `val_ap`
- reloads the saved checkpoint from that row's `artifact_path`
- runs the test split exactly once
- stores the final test result in `campaigns.tsv`

You can also finalize a specific campaign or force a specific experiment:

```bash
uv run python prepare.py finalize-campaign-test --campaign-id paper-2505-11346-r1
uv run python prepare.py finalize-campaign-test --campaign-id paper-2505-11346-r1 --experiment-num 42 --force
```

## Ledger Schema

`results.tsv` columns:

```text
campaign_id	paper_label	experiment_num	commit	val_ap	params_k	status	short_caption	description	block_id	block_label	block_run_index	block_size	block_role	seed_experiment_num	branch	log_path	artifact_path	timestamp
```

`campaigns.tsv` columns:

```text
campaign_id	paper_label	paper_url	baseline_ref	start_branch	target_experiments	idea_block_size	created_at	notes	final_test_experiment_num	final_test_commit	final_test_ap	final_test_artifact_path	final_test_timestamp
```

The root copies of `results.tsv` and `campaigns.tsv` are synced snapshots for convenience. The authoritative versions live in the shared state directory.

## Analysis

[`analysis.ipynb`](analysis.ipynb) reads the synced snapshot files and produces a combined validation-progress plot plus a campaign summary that includes the campaign-level final test result.

Run it with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

The notebook writes `progress.png`.

## Notes

- `uv` is the documented default because this repository already includes `pyproject.toml` and `uv.lock`.
- If you prefer another environment manager, install the equivalent dependencies and run the same commands inside that environment.
- PyTorch and CUDA compatibility depends on the local machine, so those packages are documented separately instead of being pinned to one CUDA wheel in `pyproject.toml`.
- CUDA is helpful but not required; the code falls back to CPU when CUDA is unavailable.

## License

MIT
