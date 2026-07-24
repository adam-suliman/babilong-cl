# Unified BABILong Continual Learning

This package runs one model continuously over a seeded permutation of:

```text
qa1, qa2, qa3, qa11, qa12, qa13
```

The primary condition is `0k`: task facts only, with no PG19 background.

## Setup

Python 3.11 and an NVIDIA driver compatible with CUDA 12.1 are expected.

```bash
git clone git@github.com:adam-suliman/babilong-cl.git
cd babilong-cl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For fish, activate with:

```fish
source .venv/bin/activate.fish
```

Verify the environment and vendored source:

```bash
CUDA_VISIBLE_DEVICES="" bash continual_learning/scripts/smoke.sh
python -m continual_learning prepare-data
bash continual_learning/scripts/gpu_smoke.sh
```

Canonical training refuses a dirty Git worktree. `--allow-dirty` is only for
smokes and development.

## Seed Contract

- `data_seed=481113` fixes the datasets and is rejected if changed in the
  canonical protocol.
- `order_seed` deterministically selects the six-task permutation. The same
  value gives every model the same order.
- `replicate_seed` controls model initialization and training randomness.
- A deterministic per-task sampler seed is derived from `replicate_seed` and
  the task name. The resolved values are recorded in every config and dry run.

For seed 48, the default order is:

```text
qa2 -> qa12 -> qa13 -> qa1 -> qa3 -> qa11
```

Compare models by changing only the model condition. Study order by changing
only `order_seed`. Estimate replicate variance by changing only
`replicate_seed`.

## Run One Condition

The condition names are `gpt2`, `gpt2-si`, `base-rmt`, `fastmem0`, and
`fastmem`.

```bash
bash continual_learning/scripts/run_one.sh base-rmt 48 48
```

The three arguments are condition, replicate seed, and order seed. The run
resumes automatically from its rolling checkpoint.

SI requires a selected coefficient. Calibrate it once on private validation
data before launching `gpt2-si`:

```bash
bash continual_learning/scripts/calibrate_si.sh
```

The fixed grid is `0.1 1.0 10.0`; selection uses final mean validation
`compare_answers` and never public test accuracy.

## Run A Suite

Inspect the exact job matrix without launching processes:

```bash
GPU_IDS="0" JOBS_PER_GPU=4 \
  bash continual_learning/scripts/run_suite.sh --dry-run
```

Launch all five conditions for one replicate/order:

```bash
GPU_IDS="0" JOBS_PER_GPU=4 \
  bash continual_learning/scripts/run_suite.sh \
  --replicate-seeds 48 \
  --order-seeds 48
```

On an A100, increase the microbatch while retaining effective batch 64:

```bash
GPU_IDS="0" JOBS_PER_GPU=4 MICROBATCH_SIZE=8 \
  bash continual_learning/scripts/run_suite.sh \
  --replicate-seeds 48 49 50 \
  --order-seeds 48
```

Protocol v3 uses one global supervised-token denominator for every 64-example
slow batch, so microbatch sizes `1`, `8`, `16`, and `32` produce the same
mathematical slow gradient. FastMem also normalizes each 32-example fast
update by its exact supervised-token count. Floating-point operation ordering
can still introduce negligible numerical differences. Values must divide
both 32 and 64.

Multiple replicates and orders can share one GPU:

```bash
GPU_IDS="0" JOBS_PER_GPU=4 \
  bash continual_learning/scripts/run_suite.sh \
  --replicate-seeds 48 49 50 \
  --order-seeds 48 49
```

Each child is an independent process. `JOBS_PER_GPU` controls concurrency;
reduce it after an OOM. The launcher requires at least 15 GB free and accounts
for the larger temporary SI checkpoint during atomic replacement. A suite
interrupt sends `SIGTERM` to children, which
checkpoint at the next slow-step boundary, and does not schedule new jobs.

## Monitoring

Training logs every 30 slow steps by default. Stage quality and CL metrics are
logged before training and after every task. Clean single-task references add
reference-normalized plasticity metrics after they complete.

```bash
bash continual_learning/scripts/tensorboard.sh
```

Then open `http://localhost:6006`. JSON remains the source of record.

## Results

```text
results/babilong_cl/
  data/qa6-0k/data-seed-481113/
  calibration/si/
  runs/<architecture>/<method>/replicate-<seed>/order-<seed>-<tasks>/
    protocol-<hash>/
      config.json
      raw.json
      summary.md
      status.json
      checkpoints/{latest,final}.pt
      tables/
      plots/
  references/<architecture>/<method>/replicate-<seed>/<identity>/<task>/
  tensorboard/<architecture>/<method>/replicate-<seed>/<order>/protocol-<hash>/
  suites/<suite-id>/
  aggregates/
```

Completed references retain predictions and metrics but delete their model and
optimizer checkpoints. CL runs retain one rolling/final checkpoint inode.

Build cross-run tables after jobs finish:

```bash
python -m continual_learning aggregate \
  --output-dir results/babilong_cl/aggregates/latest
```

## CLI

```bash
python -m continual_learning --help
python -m continual_learning dry-run --model base_rmt
python -m continual_learning run --model fastmem --replicate-seed 48 --order-seed 48
python -m continual_learning suite --help
```

The data layer exposes a provider interface for a future PG19/noisy condition.
The canonical protocol deliberately fails instead of silently generating noise,
truncating examples, or resizing GPT-2 positions.
