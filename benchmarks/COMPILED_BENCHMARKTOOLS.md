# CompiledRunner: accuracy-matched grid benchmark

This compares inference on one factor graph, not independent-model batching.
The compact backend executes the existing Gaussian rules, products and marginal
rules; neither backend uses a direct linear-system solver.

## Method

Run from `RxInfer.jl` with the three modified local packages:

```sh
julia --startup-file=no --project=benchmarks/compiled --threads=6 benchmarks/compiled_benchmark_tools.jl 64
```

- Julia 1.11.9, BenchmarkTools 1.6.3, Apple Silicon (`Sys.CPU_NAME = apple-m1`),
  32 GiB machine RAM, six Julia threads.
- 64×64 grid: 4,096 scalar unknowns, edge-list construction, Float64, diagonal
  leak 0.1, unit neighbor couplings, sinusoidal RHS, `KeepLast()`, no free energy.
- Standard reactive reference: BLAS=6, `limit_stack_depth=100`, matching the
  notebook benchmark configuration. This is not an exhaustive tuning of the
  reference's stack-depth option.
- Compiled configurations: one worker/BLAS=1, six workers/BLAS=1, six workers/BLAS=6.
  The runner itself never changes global BLAS settings.
- Common convergence gate: relative infinity-norm residual ≤1e-6, both means
  and variances stable for at least five successive sweeps (`rtol=1e-4`,
  `atol=1e-6`). Final means and variances must also match the reference at those
  tolerances, and variances must be positive.
- Calibration determines a separate sweep count per configuration, outside
  timing. Reference: 82 sweeps, residual 8.463630520035634e-7, five stable sweeps.
  All compiled configurations: 157 sweeps, residual 9.979797251642857e-7,
  16 stable sweeps. This measures equal accuracy, not equal iteration counts.
- BenchmarkTools: `evals=1`, three samples per configuration and phase per round,
  two rounds with configuration order reversed, `gctrial=true`, `gcsample=true`.
  Julia compilation is warmed. No other agent-owned Julia workload ran alongside
  these measurements; the user's idle notebook process was left untouched.
- Construction/setup includes posterior-sink attachment and detachment.
  Inference starts from a freshly constructed, initialized fixture for each
  sample (construction outside timing) and includes `KeepLast` materialization.
  End-to-end uses the public `infer` API. Accuracy checks, input generation and
  reporting are outside timing. Isolated phases need not sum exactly to API cost.

## Recorded trials

The authoritative run stores [every raw sample](compiled/results/grid64-d3cLOr/trials.json)
and [configuration/source fingerprints](compiled/results/grid64-d3cLOr/metadata.json).
The original run's temporary file was deleted at Julia exit; it is superseded by
this retained-sample rerun, not pooled with it.

Both counterbalanced rounds completed successfully; all 24 trials contain three
samples with one evaluation per sample. The following medians pool six samples
per configuration/phase (72 timed evaluations total):

| Configuration | BLAS threads | Construction/setup | Inference | End-to-end |
| --- | ---: | ---: | ---: | ---: |
| Standard | 6 | 0.329 s | 4.296 s | 4.654 s |
| Compiled, 1 worker | 1 | 0.925 s | 1.628 s | 2.537 s |
| Compiled, 6 workers | 1 | 0.921 s | 0.941 s | 1.875 s |
| Compiled, 6 workers | 6 | 0.921 s | 0.937 s | 1.869 s |

Relative to the standard reference, six compiled workers with BLAS=6 give
**4.58× faster inference and 2.49× faster end-to-end execution** on this fixture.
With BLAS=1, end-to-end speedup is 2.48×. This scalar-grid gain does not rely on
constraining BLAS.

**One-to-six-worker scaling is not linear:** at BLAS=1 it is 1.73× for inference
and 1.35× end-to-end. Compiled construction is approximately **2.80× slower**
than the reference and is effectively serial. More cores do not reduce that cost.

Round medians, shown separately to expose drift:

| Configuration | BLAS | Inference R1 / R2 | End-to-end R1 / R2 |
| --- | ---: | ---: | ---: |
| Standard | 6 | 4.279 / 4.300 s | 4.654 / 4.654 s |
| Compiled, 1 worker | 1 | 1.618 / 1.636 s | 2.536 / 2.538 s |
| Compiled, 6 workers | 1 | 0.941 / 0.934 s | 1.876 / 1.875 s |
| Compiled, 6 workers | 6 | 0.935 / 0.939 s | 1.866 / 1.873 s |

These are small-sample descriptive comparisons, not confidence intervals. This is
one scalar grid at matched accuracy, not a speedup guarantee for other models or
an extrapolation to the million-variable case.

Allocation pressure remains: the reference inference trial allocates
1,522,849,248 bytes in 35,223,030 allocations, versus 1,249,592,160 bytes in
49,654,015 allocations for six compiled workers. Median inference GC time is
399.68 ms for the reference, 65.53 ms for one compiled worker and 182.32 ms for
six workers/BLAS=6. Compaction reduced retained graph state; it did not eliminate
temporary wrappers, allocation overhead, task overhead or result materialization.

## Separate capacity result

The actual 1000×1000 grid (one million unknowns) passed with 158 sweeps, residual
9.8859e-7, 17 stable mean/variance sweeps, 7.7708 GiB peak process RSS and
5.2138 GiB final live heap. Six workers, BLAS=1, heap hint=4G, `KeepLast()`.
This was a capacity/correctness probe under machine memory pressure, not a timing
benchmark. The default reactive backend was not run at this size, so no
million-variable speedup is claimed. See [the implementation record](COMPILED_BACKEND_PROGRESS.md)
for revision lineage, failed capacity attempts and remaining compatibility work.
