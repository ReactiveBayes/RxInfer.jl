# Experimental multicore runner: measurements

`MulticoreRunner` schedules ready computations within **one inference graph**.
It is not a dataset splitter, multiple independent fits, or a Gaussian-only
backend. The existing RxInfer model and inference API are used unchanged except
for `options = (runner = MulticoreRunner(),)`.

## Reproduce

### BenchmarkTools protocol (supersedes the manual timing protocol)

The scripts now share `multicore_benchmark_tools.jl` and use BenchmarkTools
`@benchmarkable`/`run`, not hand-written `@elapsed` loops. The historical tables
below were collected before this change; they are exploratory measurements,
not BenchmarkTools-validated results. Do not mix them with new trials.

The completed [BenchmarkTools remeasurement](MULTICORE_BENCHMARKTOOLS.md) tests
the notebook's 64 × 64, leak-0.1 grid with default BLAS=6: standard median
5.705 s, six-worker median 9.244 s. It still shows no acceleration, and the wave
schedule has a larger residual after the same 100 iterations.

A subsequent [matched larger-grid test](MULTICORE_LARGE_GRIDS.md), using an
equivalent sparse edge-list encoding and 20 iterations at both sizes, compared
standard BLAS=6 against wave-runner BLAS=1. At 200 × 200 the medians were 20.24 s
standard, 30.25 s with one worker and 30.41 s with six workers. The larger grid
still did not produce useful worker scaling. A 1000 × 1000 graph was not run
because its projected retained memory alone exceeds the host's physical RAM.

The common protocol uses interpolated arguments, a fresh inference per
evaluation (`evals=1`), five samples per round and two rounds in forward/reverse
configuration order. Each configuration is warmed and checked before timing;
BenchmarkTools also warms its generated measurement function. Correctness
checks run outside timing after each trial. Graph construction, inference and
GC during inference are included. A wrapper returns `nothing` so BenchmarkTools
does not retain the first full graph throughout a trial. Both `gctrial` and
`gcsample` are true for every configuration. All samples are retained; summaries
include median, minimum, quartiles, GC time, allocations and each round's median.
The parameter meanings and interpolation rules are documented in the
[BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).

BLAS is left at its process default unless explicitly selected with
`RXINFER_BENCHMARK_BLAS_THREADS`. By default that selection is identical for
every worker count. To compare a tuned wave runner against the standard
runner's normal BLAS configuration, set
`RXINFER_BENCHMARK_RUNNER_BLAS_THREADS` separately (for example, standard BLAS=6,
wave BLAS=1). Settings change outside timing, are printed for each trial and
stored in the raw trial-group tags, and the original BLAS setting is restored
on exit. Use `RXINFER_BENCHMARK_SAMPLES` and
`RXINFER_BENCHMARK_ROUNDS` to set sampling, and `RXINFER_BENCHMARK_OUTPUT` to save
raw BenchmarkTools trials to a new JSON file. Keep the Julia version, startup
options, heap hint, data, iteration count and each configuration's BLAS setting
fixed across rounds. BenchmarkTools cannot remove thermal/load effects or turn an
equal-iteration comparison into a time-to-equal-accuracy result.

Use a Julia-compatible environment containing BenchmarkTools and both local
packages. The checked-in benchmark Manifest currently targets Julia 1.12.5;
do not use it unchanged with Julia 1.11. The September 5 remeasurement uses a
temporary Julia 1.11.9 environment seeded from the working RxInfer Manifest,
with BenchmarkTools 1.6.3 added offline, leaving repository manifests untouched.

From the directory containing the local RxInfer.jl and ReactiveMP.jl checkouts,
with both local packages developed in the selected Julia environment:

```sh
julia --startup-file=no --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_scaling.jl 256 128 20
julia --startup-file=no --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_grid.jl 32 40
julia --startup-file=no --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_linear_system.jl 64 100 0.1
julia --startup-file=no --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_rslds.jl 100 30 3
```

The new linear-system script loads the actual notebook model with its dense
edge enumeration. It uses the notebook's degree-plus-leak grid matrix, a
deterministic sinusoidal RHS and `KeepLast` rather than history/plots. It reports
relative residuals and errors against a direct sparse solve outside timing.
The older `multicore_grid.jl` has diagonal 5 everywhere; it is a different,
more strongly damped test, not the notebook's leak-0.1 matrix.

The RSLDS script also needs the sibling RxInferExamples.jl checkout. It loads
the model and custom rules directly from its existing notebook, without plots.
It substitutes the standard-library seeded RNG for StableRNGs, and changes
the horizon and inference options; it does not rewrite the model or its rules.

## Coupled multivariate model

Measured on this machine with Julia 1.11.9, `Sys.CPU_NAME == "apple-m1"`, six
available Julia threads, and single-threaded BLAS. Each observation has a
256-dimensional latent state, and all 128 observations share an inferred
precision: 32,768 latent state dimensions in one coupled graph. Twenty VMP
iterations; five repetitions after warm-up. Times include model construction
and inference, exclude first-call compilation, and use GC before each sample.
The benchmark checks both the mean and variance of the final shared-precision
posterior against the standard runner (`rtol = 1e-10`).

| Runner | Median seconds | Speedup over standard |
| --- | ---: | ---: |
| Standard reactive | 2.9814 | 1.00× |
| Multicore, 1 worker | 2.9518 | 1.01× |
| Multicore, 2 workers | 1.6175 | 1.84× |
| Multicore, 4 workers | 0.9428 | 3.16× |
| Multicore, 6 workers | 0.7638 | 3.90× |

Raw samples in seconds:

```text
standard: 3.218989500 3.002944334 2.977246875 2.974699083 2.981353583
1 worker: 3.004523875 2.935701500 2.967324709 2.945791083 2.951802958
2 workers: 2.091633458 1.659805833 1.610249417 1.617528584 1.613807041
4 workers: 0.943423416 0.958768250 0.940453875 0.942806041 0.938545458
6 workers: 0.766572916 0.761511166 0.784598875 0.748679333 0.763794208
```

The final repeat passed both posterior checks but exposed six-worker
variability:

| Runner | Final-repeat median seconds | Speedup over standard |
| --- | ---: | ---: |
| Standard reactive | 3.0278 | 1.00× |
| Multicore, 1 worker | 3.0135 | 1.00× |
| Multicore, 2 workers | 1.6328 | 1.85× |
| Multicore, 4 workers | 0.9491 | 3.19× |
| Multicore, 6 workers | 1.6944 | 1.79× |

```text
standard: 3.239511708 3.036027417 3.027782958 3.010023834 3.021739625
1 worker: 3.091607917 3.020674791 3.013491833 2.983085417 2.992724750
2 workers: 1.682517959 1.630830666 1.632836000 1.635308625 1.614792000
4 workers: 0.943080250 0.942972667 0.949067542 1.057243333 1.340205542
6 workers: 1.592246333 1.711304708 1.717358542 1.693034208 1.694412417
```

The strongest repeatable result is about **3.2× with four workers**; the 3.9×
six-worker result was not stable across runs. The cause of this variability
has not been isolated. These results do **not** establish linear scaling.
An earlier run
with twice as many latent vectors had highly variable six-worker measurements
(1.46 s minimum versus 4.30 s median) as well. All reported medians use every
sample in their run; no slow samples were discarded.

## Scalar grid: no speedup

A 32 × 32 grid using the linear-systems example's `GaussianCoupling` rules,
40 iterations, and five warmed repetitions:

| Runner | Median seconds |
| --- | ---: |
| Standard reactive | 0.5265 |
| Multicore, 1 worker | 0.8859 |
| Multicore, 2 workers | 0.9031 |
| Multicore, 4 workers | 0.8229 |
| Multicore, 6 workers | 0.8364 |

All worker counts gave exactly matching wave-schedule posteriors. Means and
variances agreed with standard execution to `atol = 1e-7`. The scalar rules
are too cheap to recover the queuing, snapshotting and serial-delivery costs.
Increasing worker count alone does not address this bottleneck.

## RSLDS: no speedup

The existing two-dimensional, two-state RSLDS model, with 100 time steps and
30 iterations, was timed over three warmed repetitions using the default
adaptive runner configuration:

| Runner | Median seconds |
| --- | ---: |
| Standard reactive | 0.7364 |
| Multicore, 1 worker | 0.7968 |
| Multicore, 2 workers | 1.0525 |
| Multicore, 4 workers | 1.0908 |
| Multicore, 6 workers | 1.1691 |

This is a regression, not an acceleration. The adaptive threshold reduces task
creation but does not eliminate the overhead on this workload. The wave runner
produced exactly matching posterior values and free-energy sequences across
worker counts. Its final free energy was 326.5496 versus 325.6653 for the
standard schedule at the same iteration count. No time-to-equal-accuracy gain
is claimed. Larger horizons and state dimensions remain to be benchmarked.

## Scheduling and limits

Workers compute captured inputs; all reactive delivery remains on the calling
task in emission order. This wave schedule differs from the standard runner's
depth-first schedule. Worker counts have matching outputs in the tested models,
but finite-iteration beliefs, free-energy trajectories and convergence rates
can differ from standard execution. In particular, the RSLDS example does not
have identical finite-iteration free energy under the two schedules. Comparing
seconds for the same iteration count is not a time-to-equal-accuracy result.

Threading cheap scalar rules can cost more than evaluating them. By default,
waves need at least 32 eligible jobs and 200 microseconds of estimated work
before spawning tasks. The runner calibrates using up to nine actual updates
per job type, excludes the first call from timing, and never evaluates a
sample twice. `min_work_ns = 0` disables calibration for controlled experiments.

Callbacks, annotations, required factor-node state, mutable approximation
metadata and unknown mappings take a conservative serial path. Custom parallel
rules must treat distributions as read-only and must not mutate shared global
state. Model creation, reactive delivery, equality-chain caching and free-energy
scoring are still serial. Dependency chains, narrow waves and memory bandwidth
limit speedups; this is not a universal acceleration guarantee.

## Verification

The BenchmarkTools harness has 48 passing checks for evaluation/sample counts,
warm-up behavior, GC settings, result checks, per-configuration BLAS selection
and BLAS restoration on both normal and exceptional exits. Small smoke runs passed for all four scripts, including
the notebook-loaded linear system and RSLDS; raw-trial JSON saving was exercised.
Those smoke timings are not performance measurements.

The final targeted TestItemRunner checks passed with Julia 1.11.9:

- Six threads: 108 RxInfer checks (84 new runner checks and 24 existing options
  checks).
- One thread: 105 RxInfer checks (the three thread-dependent assertions are
  skipped).
- 5,487 existing ReactiveMP checks for random variables, local marginal
  clusters and functional dependencies.

The new tests exercise conjugate Gaussian/Gamma inference, categorical HMMs,
loopy Gaussian grids, structured chains, missing data, nonlinear Delta nodes,
streaming auto-updates, ordered delivery, worker exceptions, state cleanup,
mutable-state fallback, and exactly-once adaptive calibration. Model tests
force parallel dispatch to ensure the adaptive fallback cannot mask threading
errors. This is a targeted regression run, not the entire package test suite.
Both local packages were developed into a temporary offline test environment;
no repository Project.toml or Manifest.toml was changed.

All edited Julia files passed the repositories' JuliaFormatter configuration,
and tracked changes passed `git diff --check`.

The current implementation demonstrates a general parallel execution path and
a substantial gain for expensive message rules. A gain for large scalar grids
or small-state RSLDS, and near-linear scaling in general, are **not established**.
