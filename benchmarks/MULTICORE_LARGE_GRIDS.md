# Larger grids and per-runner BLAS settings

September 5, 2026. This comparison uses standard execution with BLAS=6 and
the wave runner with BLAS=1, as requested. One-worker/BLAS=1 execution is
included to distinguish changing the worker count from changing BLAS.

The completed 200 × 200 run still shows no useful multicore scaling: standard
execution takes 20.24 s, versus 30.41 s with six workers and BLAS=1. Six workers
are also slightly slower than one worker at the same BLAS setting. The roughly
1.50× regression is essentially unchanged from the matched 64 × 64 test.

## Representation and memory preflight

A side-m grid has m² unknowns, so the notebook's dense Float64 coefficient
matrix needs 8m⁴ bytes. That is 12.8 GB at 200 × 200 and 8 TB at 1000 × 1000.
This test instead uses a generic edge-list linear-system model with the same
Gaussian priors and GaussianCoupling factors. Edge order matches the notebook's
ascending i/j traversal. The general MulticoreRunner is unchanged; there is no
special grid kernel and no splitting into independent inference problems.

On an 8 × 8 grid, the edge-list and notebook versions produced exactly matching
means and variances after ten iterations, separately under standard, one-worker
and six-worker execution.

Memory probes used two iterations and KeepLast. Retained heap is measured after
full GC while the result graph remains live; RSS includes runtime/compiler and
transient memory. Values are decimal GB.

| Grid | Standard added live heap | Six-worker added live heap | Process peak RSS through six-worker probe |
| --- | ---: | ---: | ---: |
| 64 × 64 | 0.160 | 0.161 | 1.241 |
| 96 × 96 | 0.371 | 0.372 | 1.628 |
| 200 × 200 | 1.607 | 1.617 | 3.783 |

The smaller probes used a 1536M heap hint; 200 × 200 used 3G. These were memory
preflights, not comparable performance timings. Extrapolating the 200 × 200
retained graph to one million unknowns gives roughly 40 GB before runtime and
transient overhead. The host has 32 GiB physical RAM. A 1000 × 1000 run was
therefore not attempted; its performance is unknown, not an extrapolated timing.

## Matched performance protocol

- Both 64 × 64 and 200 × 200 use edge lists, leak 0.1, `b = sin.(1:side^2)`,
  20 iterations, KeepLast and the same 3G heap hint (a GC preference, not a cap).
- Julia 1.11.9, BenchmarkTools 1.6.3, `--startup-file=no`, six Julia threads.
  Standard BLAS=6; one-worker and six-worker wave runners both use BLAS=1.
- Shared BenchmarkTools harness: evals=1, three samples per round, two rounds
  in forward/reverse order, gctrial=true, gcsample=true. All samples retained.
- Timings include graph construction and inference. Edge-list construction and
  correctness checks are excluded. Each inference creates a fresh graph.
- Preflight and post-trial checks require finite means, positive finite
  variances, repeatability and exact wave-runner agreement across worker counts.
  Residuals are reported; these are not time-to-equal-accuracy measurements.

Do not compare these seconds directly with the earlier dense 64 × 64,
100-iteration, BLAS=6-only benchmark: workload representation, iteration count
and heap settings differ. Only compare sizes within this matched protocol.

## Completed measurements

| Grid | Standard / BLAS=6 median | 1 worker / BLAS=1 median | 6 workers / BLAS=1 median |
| --- | ---: | ---: | ---: |
| 64 × 64 | 1.418088 s | 2.050709 s | 2.138474 s |
| 200 × 200 | 20.239553 s | 30.248104 s | 30.408070 s |

| Grid / configuration | Minimum | Q25–Q75 | Round 1 median | Round 2 median | Median GC seconds | Allocated bytes* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 / standard, BLAS=6 | 1.363121 | 1.375514–1.464987 | 1.466818 | 1.375125 | 0.123138 | 678,569,440 |
| 64 / 1 worker, BLAS=1 | 2.035162 | 2.047755–2.054684 | 2.046806 | 2.055973 | 0.168794 | 894,381,200 |
| 64 / 6 workers, BLAS=1 | 2.135012 | 2.137623–2.141534 | 2.137574 | 2.139176 | 0.167305 | 901,929,488 |
| 200 / standard, BLAS=6 | 19.935406 | 20.096977–20.417550 | 20.091999 | 20.434335 | 4.713178 | 6,760,183,104 |
| 200 / 1 worker, BLAS=1 | 29.678718 | 29.903663–30.350179 | 29.808664 | 30.364389 | 6.687397 | 8,885,732,512 |
| 200 / 6 workers, BLAS=1 | 30.292818 | 30.385563–30.716832 | 30.382204 | 30.420502 | 6.692839 | 8,958,604,320 |

*Minimum allocation estimate across the trials, not peak resident memory.
Process peak RSS was 1.746 GB through the 64 × 64 run and 4.377 GB through the
200 × 200 run. The 3G heap hint was not a hard cap. Both sizes' full inference
graphs fit; allocation totals include collected and reused memory.

| Grid | Standard relative residual | Wave-runner relative residual |
| --- | ---: | ---: |
| 64 × 64 | 1.856468e-3 | 1.998413e-2 |
| 200 × 200 | 2.052235e-3 | 2.032861e-2 |

Residuals are `norm(A * mean_x - b, Inf) / norm(b, Inf)`, evaluated directly
from the edge list without a dense A. Wave-runner means and variances matched
exactly across worker counts and repeated checks. The wave schedule is less
converged at this fixed iteration count; no equal-accuracy speedup is claimed.

At 200 × 200, 4,783,964 numerical jobs were dispatched to workers and 36 ran
serially for calibration, across 40 waves. That is roughly 119,600 jobs per
wave: lack of ready jobs is not preventing worker dispatch. These counts do
not mean most runtime is parallel; snapshotting, reactive publication,
equality-chain caching and model construction remain serial. Increasing grid
size scales the per-message overhead as well as the numerical work. The stable
regression and almost unchanged one-to-six-worker runtime are consistent with
that limitation, but no detailed profile was collected in this test.

### Raw samples

Seconds, round 1 followed by round 2; no samples discarded:

```text
64 standard BLAS6: 1.459492917 1.468038917 1.466818083 1.363121000 1.376682459 1.375125167
64 wave1 BLAS1: 2.046806084 2.035161959 2.050602250 2.065714833 2.055973459 2.050814875
64 wave6 BLAS1: 2.137573875 2.135012334 2.142336083 2.142319417 2.137770959 2.139176333
200 standard BLAS6: 20.091998875 19.935406375 20.111911833 20.975702084 20.434335292 20.367194458
200 wave1 BLAS1: 29.808663500 30.307548042 29.678718083 30.364388667 30.405882041 30.188660666
200 wave6 BLAS1: 31.451041792 30.292817750 30.382204333 30.395637708 30.815608500 30.420502375
```

## Artifacts and reproduction

- [Edge-list model, equivalence checks and memory probes](/private/tmp/rxinfer-large-grid.EUDZBt/probe.jl)
- [200 × 200 memory preflight](/private/tmp/rxinfer-large-grid.EUDZBt/memory200.jl)
- [Matched-size BenchmarkTools driver](/private/tmp/rxinfer-large-grid.EUDZBt/benchmark.jl)
- [64 × 64 raw trials](/private/tmp/rxinfer-large-grid.EUDZBt/grid64-blas6-vs-blas1.json)
- [200 × 200 raw trials](/private/tmp/rxinfer-large-grid.EUDZBt/grid200-blas6-vs-blas1.json)

The temporary Julia-compatible environment from the earlier BenchmarkTools
measurement is reused. Both RxInfer and ReactiveMP are the modified local
checkouts. Run from the workspace root:

```sh
RXINFER_BENCHMARK_SAMPLES=3 RXINFER_BENCHMARK_ROUNDS=2 RXINFER_BENCHMARK_BLAS_THREADS=6 RXINFER_BENCHMARK_RUNNER_BLAS_THREADS=1 julia --startup-file=no --project=/private/tmp/rxinfer-benchmarktools.NQeR2S --threads=6 --heap-size-hint=3G /private/tmp/rxinfer-large-grid.EUDZBt/benchmark.jl 20
```

The driver refuses to overwrite its raw output files; choose new filenames in
a copy of the driver for another run. The shared harness now has 48 passing
checks, including per-configuration BLAS settings during measured evaluations
and restoration after failures. No runner implementation changes were made.
