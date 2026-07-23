# QA6 0k Protocol

## Scope

The experiment measures semantic continual learning over six bAbI-derived
location-answer tasks. It is not a long-context BABILong result because it uses
no PG19 distractors. RMT recurrence is exercised only by fitting QA3 rows over
512 tokens.

Every run trains one model sequentially with no replay, model
reinitialization, early stopping, or public-test checkpoint selection. AdamW
weights and moments persist through all six tasks. A task-local linear
scheduler restarts at each task boundary.

## Data

All conditions share one immutable data manifest.

Protocol-v1 canonical manifest SHA256:
`600b4a4791b5d213d56c3226e7553a60ef5401c2a1ed80f40ead439a51722292`.

| Tasks | Training source |
|---|---|
| QA1, QA2, QA3 | `RMT-team/babilong-train-5k-samples`, revision `b3513ef7...`, `0k` JSON |
| QA11, QA12, QA13 | bAbI `en-10k` train scenarios parsed with the exact released RMT `TaskDataset` |
| All public evaluations | `RMT-team/babilong`, revision `ee0d588...`, `0k` JSON |
| SI validation | disjoint bAbI `en-valid-10k` scenarios, 400 rows per task |

Generated QA11-13 rows exclude exact normalized overlaps with public
evaluation. Validation excludes overlaps with both train and public
evaluation. Source members, scenario selections, file hashes, tokenizer
revision, and token statistics are recorded.

Serialization and loss masking reproduce the released trainer:

```text
input + question + GEN + target + EOS
```

Loss is applied to answer and EOS predictions using the released
`labels_mask` convention.

Stock GPT-2 is fail-closed at 1,024 positions. In the canonical manifest,
QA3 has 5,000 source rows, 4,946 fitting rows, and 54 excluded rows. The
largest included QA3 row is 1,022 tokens. The same fitting subset is used by
all models. Every public evaluation row fits.

## Models

- `gpt2`: fully fine-tuned GPT-2 small, native positions, no memory tokens.
- `gpt2-si`: the same model and optimizer with Synaptic Intelligence.
- `base_rmt`: GPT-2 plus 16 learned memory tokens and the byte-identical
  released `MemoryCell` and `RecurrentWrapper`.
- `fastmem0`: FastMem-RMT with `fast_lr=0`, the mechanism control.
- `fastmem`: FastMem-RMT with `fast_lr=0.005`.

RMT uses 512-token segments, at most two segments, and full segment BPTT
(`k2=-1`). The exact wrapper and parser are under
`third_party/rmt_babilong_release`; provenance hashes are checked by the smoke
suite. Local orchestration and FastMem are adaptations, not byte-identical
upstream trainer code.

FastMem has a slow learned initializer owned by AdamW and an active fast
parameter excluded from AdamW. Training uses:

```text
fast + initializer - initializer.detach()
```

The active state receives two manual 32-example updates per 64-example slow
step. It resets from the learned initializer at epoch, task, and evaluation
boundaries. Evaluation uses the learned initializer. `fastmem0` makes the
same update attempts without changing active memory.

## Optimization

- 3,001 slow AdamW steps per task (`3,000` is retained as the historical
  inclusive-iteration configuration).
- Effective slow batch 64; microbatch 1 by default; full batches only.
- AdamW weight decay `0.01`, gradient clipping `1.0`.
- Learning rate `3e-5` for QA3 and `1e-5` for all other tasks.
- Linear task-local scheduler with 300 warmup steps.
- FP32 default.
- Persistent model, learned memory, AdamW moments, RNG, and cumulative
  counters.

Task data are reshuffled deterministically each epoch. Tail rows that do not
form a full 64-example batch are dropped for that epoch. Runs record source,
fitting, effective, and dropped row counts.

## SI Baseline

SI matches the equations and hook timing of the repository used for the paper's
incremental MQAR experiment: `epsilon=0.1`, `decay=1`, nonnegative importance,
and slow optimizer-owned parameters only. It captures the final accumulated,
clipped gradient immediately before AdamW and accumulates
`-gradient * parameter_delta`.

`si_lambda` is selected from `{0.1, 1.0, 10.0}` using one fixed seed/order and
private validation only. The chosen value is then fixed for public-test runs.
See `strategies/PROVENANCE.json`.

## Evaluation And CL Metrics

All six public tasks are evaluated before training and after each task,
producing a `7 x 6` matrix. Deterministic short generation is used. The
official wrapper returns generated-only token IDs under the pinned
Transformers version; evaluation truncates these IDs at the first EOS token
before decoding the answer.

Primary quality is official BABILong `compare_answers`; strict normalized exact
match is secondary. Reports include:

- current-task and mean learning accuracy;
- final all-task and old-task accuracy;
- learning-to-final and best-to-final forgetting;
- backward transfer and forward transfer;
- clean single-task reference accuracy;
- intransigence and reference-normalized plasticity.

Each single-task reference reconstructs the run's exact initial model/memory
state, starts fresh AdamW, and trains only one task with the matched task
budget and sampler seed. References are cached by model, replicate, task,
data hash, and protocol; task-order sweeps reuse them.

FastMem-specific claims are allowed only when nonzero FastMem beats both Base
RMT and `fastmem0` with matched backbone, memory size, data, seed, order,
precision, batch policy, budgets, and optimizer-step counts.

## Checkpointing

The rolling checkpoint stores model, persistent AdamW, scheduler, SI state,
active FastMem state, sampler cursor, RNG states, progress, parameter manifest,
protocol hash, initial-model hash, and data-manifest hash. Incompatible state
fails closed. `SIGINT`/`SIGTERM` requests a checkpoint at the next complete
slow step. The final checkpoint is a hardlink to the rolling state when the
filesystem supports it.

JSON and CSV are authoritative. TensorBoard and generated figures are
monitoring and presentation layers.
