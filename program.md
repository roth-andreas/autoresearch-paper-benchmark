# autoresearch workflow

Use this workflow when you are asked to run a paper-driven campaign in this repository.

## Goal

Transfer ideas from one research paper into [`train.py`](train.py) and evaluate them on the fixed benchmark defined by [`prepare.py`](prepare.py).

## Important Rules

- Treat [`prepare.py`](prepare.py) as the fixed benchmark harness.
- Put model and training changes in [`train.py`](train.py).
- Use validation AP to choose what to try next.
- Do not use the test split during the campaign.
- Run the test split once, at the end, through `prepare.py finalize-campaign-test`.

## Benchmark Contract

The benchmark is fixed:

- dataset: `Peptides-func`
- split: `train / val / test`
- experiment metric: validation AP
- time budget per experiment: `TIME_BUDGET`
- parameter cap: `MAX_PARAMS`

## Start Of A Campaign

When starting a new paper:

1. Read [`README.md`](README.md), [`prepare.py`](prepare.py), and [`train.py`](train.py).
2. Read the paper carefully enough to identify ideas that can realistically transfer.
3. Initialize local tracking if needed:

```bash
python prepare.py init-results
```

4. Create or activate a campaign:

```bash
python prepare.py bootstrap-campaign --paper-label "<paper title>" --paper-url <paper url>
```

5. Confirm the active campaign:

```bash
python prepare.py show-campaign
```

## Experiment Loop

Inside one campaign:

1. Run a baseline first.
2. Choose one paper-motivated idea.
3. Implement that idea in [`train.py`](train.py).
4. Run one experiment with `prepare.py run-and-log`.
5. Inspect the validation result.
6. Decide the next change based on the paper and the newest validation evidence.
7. Repeat until the campaign reaches its target number of experiments.

The campaign should feel like iterative research, not a blind hyperparameter grid.

Good changes:

- central architectural ideas from the paper
- paper-motivated normalization or aggregation changes
- readout changes suggested by the paper
- small local tuning once a promising direction appears

Bad changes:

- unrelated architecture churn
- using the test set to guide decisions
- planning all 15 experiments in advance and running them blindly

## Logging

Run experiments through the helper:

```bash
python prepare.py run-and-log \
  --short-caption "<caption>" \
  --description "<what changed and why>"
```

Rules:

- keep `short_caption` readable and short
- write `description` so a new user can understand what changed
- never hand-edit `results.tsv`
- each `run-and-log` call should correspond to one real experiment

If training crashes or times out before producing a usable checkpoint, the run should still be logged as a crash row.

## Final Test

After the campaign reaches its target number of experiments:

```bash
python prepare.py finalize-campaign-test
```

That command picks the recoverable experiment with the best validation AP and evaluates the test split once.

## Practical Guidance

- Prefer simple, paper-aligned changes over large rewrites.
- Keep experiments easy to describe.
- If a direction is clearly weak, move on instead of over-tuning it.
- If one direction is promising, spend a few runs refining it.
- Preserve recoverability: the best model in each run should be checkpointed by [`train.py`](train.py).

## What `prepare.py` Owns

- dataset loading
- train/validation/test split access
- metric computation
- experiment timeout
- local campaign bookkeeping
- artifact directory creation
- final one-shot test evaluation

## What `train.py` Owns

- model architecture
- optimizer and learning-rate choices
- regularization
- checkpointing the best validation model
- optional inner candidate selection inside one experiment

## Persistence

Do not stop after one weak experiment.
Keep iterating until the requested number of experiments is complete, unless the user explicitly redirects you.
