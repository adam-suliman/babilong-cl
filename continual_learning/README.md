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

The default v4 protocol follows the incremental-AR cadence. Every condition
sees 6,002 physical 32-example minibatches per task. GPT-2, SI, and Base RMT
take 6,002 slow steps; FastMem/FastMem0 average two minibatches and take 3,001
slow steps. The completed equal-slow-step protocol remains available as
`configs/qa6_0k_v3.json` and must not be mixed with v4 results.

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

On an A100 or H100, increase the microbatch while retaining the configured
physical and effective batch sizes:

```bash
GPU_IDS="0" JOBS_PER_GPU=4 MICROBATCH_SIZE=8 \
  bash continual_learning/scripts/run_suite.sh \
  --replicate-seeds 48 49 50 \
  --order-seeds 48
```

Protocol v4 uses one global supervised-token denominator for every effective
slow batch, so microbatch sizes `1`, `8`, `16`, and `32` produce the same
mathematical slow gradient. FastMem also normalizes each 32-example fast
update by its exact supervised-token count. Floating-point operation ordering
can still introduce negligible numerical differences. Values must divide both
32 and 64.

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

## Two-H100 Priority Run

The tracked profile uses only GPUs 0 and 1, two jobs per GPU, replicate seeds
49/50, and order seeds 230/806:

```bash
cat continual_learning/configs/h100_two_gpu_priority.env
bash continual_learning/scripts/run_h100_two_gpu_ar_analog.sh dry-run
```

Run the staged workflow in tmux:

```bash
tmux new -s babilong-v4
bash continual_learning/scripts/run_h100_two_gpu_ar_analog.sh all
```

It queues the Base RMT/FastMem0/FastMem control triad first, finishes CL without
references, computes every unique clean reference once, attaches the cached
references, and aggregates the results. Rerunning resumes compatible
checkpoints and skips verified completed work. If time remains, run the
additional order with:

```bash
bash continual_learning/scripts/run_h100_two_gpu_ar_analog.sh extension
```

## Monitoring

Training logs every 30 slow steps by default. Stage quality and CL metrics are
logged before training and after every task. Clean single-task references add
reference-normalized plasticity metrics after they complete.

```bash
bash continual_learning/scripts/tensorboard.sh --port 6009
```

The helper defaults to `results/babilong_cl_v4_ar/tensorboard`. Set
`RESULTS_ROOT` or `TENSORBOARD_DIR` to inspect another protocol tree.

Then open `http://localhost:6009`. JSON remains the source of record.

## Results

```text
results/babilong_cl_v4_ar/
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
  --output-dir results/babilong_cl_v4_ar/aggregates/latest
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
