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

- `results.tsv`
- `campaigns.tsv`
- `results/current_campaign.json`
- `progress.png`

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

The environment manager does not matter. `uv`, `conda`, `venv`, and similar tools are all acceptable as long as `python train.py` and `python prepare.py ...` run in the same GPU-capable environment.

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

Bootstrap a new paper campaign from the neutral baseline:

```bash
uv run python prepare.py bootstrap-campaign \
  --paper-label "MIMO Graph Conv" \
  --paper-url https://arxiv.org/abs/2505.11346 \
  --target-experiments 50
```

By default, `bootstrap-campaign` starts from `neutral-baseline`. If you want to bootstrap from a different trusted ref, pass `--baseline-ref <git-ref>`.

If `neutral-baseline` does not exist yet, create it explicitly from a trusted branch or commit:

```bash
uv run python prepare.py bootstrap-campaign \
  --paper-label "MIMO Graph Conv" \
  --paper-url https://arxiv.org/abs/2505.11346 \
  --create-baseline-from neutral-baseline \
  --target-experiments 50
```

The command prints a summary including:

- `campaign_id`
- `start_branch`
- `baseline_ref`
- `worktree_path`

Open that worktree and continue there.

## Using An AI Agent

[`program.md`](program.md) is meant to be a lightweight agent skill. Open your coding agent in this repository, point it at `program.md`, and ask it to start with setup before it begins running experiments.

A good starter prompt is:

```text
Hi have a look at program.md and let's kick off a new experiment! Let's do the setup first.
The models should follow the ideas of this paper: https://arxiv.org/abs/2505.11346
```

If your agent supports permission modes, restricted mode is a good default while you are getting comfortable with the workflow.

The expected flow is:

1. The agent reads `README.md`, `prepare.py`, `train.py`, and `program.md`.
2. The agent bootstraps or activates a paper campaign.
3. The agent runs the baseline first.
4. The agent iterates on `train.py` with paper-aligned changes.
5. The agent logs one validation result per experiment.
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
Hi have a look at program.md and let's kick off a new experiment! Let's do the setup first.
The models should follow the ideas of this paper: https://arxiv.org/abs/2505.11346
```

```text
Hi have a look at program.md and let's kick off a new experiment! Let's do the setup first.
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
2. Implement one paper-driven change in [`train.py`](train.py).
3. Run `uv run python train.py`.
4. Read the reported `Selected candidate`, `Params k`, `Artifact path`, and `Final val`.
5. Log one experiment row with `prepare.py log-result`.
6. Repeat until the campaign reaches its target count.
7. Finalize the holdout test once.

Example logging command:

```bash
uv run python prepare.py log-result \
  --commit b09960f \
  --val-ap 0.424 \
  --params-k 49.9 \
  --status keep \
  --short-caption "LMGC p35 b180" \
  --description "LMGC tanh mixer with patience 35 and batch size 180" \
  --log-path run-exp44.log \
  --artifact-path /path/to/artifact
```

Rules:

- `val_ap` is the only per-experiment score stored in `results.tsv`
- `artifact_path` is required for every non-crash row
- crashes should still be logged with status `crash`, `val_ap 0`, and `params_k 0`
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
campaign_id	paper_label	experiment_num	commit	val_ap	params_k	status	short_caption	description	branch	log_path	artifact_path	timestamp
```

`campaigns.tsv` columns:

```text
campaign_id	paper_label	paper_url	baseline_ref	start_branch	target_experiments	created_at	notes	final_test_experiment_num	final_test_commit	final_test_ap	final_test_artifact_path	final_test_timestamp
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
