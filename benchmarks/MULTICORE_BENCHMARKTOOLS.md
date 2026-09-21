# Multicore runner: BenchmarkTools remeasurement

Measured September 5, 2026. The current runner does **not** accelerate the
tested scalar linear-system grid. Six workers take 1.62 times as long as
standard execution, and are slower than the wave runner with one worker.
These results replace the manual timing probes for this configuration; do not
mix samples or speedup ratios from the two protocols.

## Workload and protocol

- The actual `linear_system_model` is loaded from the linear-systems notebook,
  including its dense upper-triangle edge enumeration.
- A 64 × 64 grid (4,096 unknowns), degree-plus-0.1 diagonal, neighbor entries
  -1, deterministic `b = sin.(1:4096)`, 100 iterations, `KeepLast`.
- Timed scope: a fresh `infer` call, including graph construction, inference,
  result collection and GC during the call. Matrix/RHS construction, direct
  sparse reference solve, residual checks, SPD validation, history and plotting
  are excluded. This is not an untouched whole-notebook timing.
- Julia 1.11.9, BenchmarkTools 1.6.3, CPU reported as `apple-m1`, six Julia
  threads, `--startup-file=no`. BLAS stays at its default of six threads for
  **every** configuration. The runner does not change BLAS.
- The same `--heap-size-hint=1536M` is used throughout. It is a GC preference,
  not a memory cap. Process peak RSS was 1.826 GB.
- `@benchmarkable` with interpolated inputs; `evals=1`, `gctrial=true`,
  `gcsample=true`. Two rounds, three samples per round, six samples per
  configuration. Orders: standard/1/2/4/6, then 6/4/2/1/standard.
- Full-size preflight warm-up and correctness checks for every configuration;
  BenchmarkTools also warms its measurement function. Checks after each trial
  are outside timing. A result-discarding wrapper prevents BenchmarkTools
  retaining the first full graph for the lifetime of a trial.
- All six samples contribute to the summaries. No trimming or outlier removal.
  The trial time ceiling is 3,600 seconds so the default five-second budget
  cannot silently give slower configurations fewer samples.

The shared implementation is [multicore_benchmark_tools.jl](multicore_benchmark_tools.jl),
using the [documented BenchmarkTools controls](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).
It is also used by the diagonal-5 grid, multivariate and RSLDS scripts. Those
other workloads have only had small smoke checks under the new harness so far;
their historical performance tables have **not** been revalidated here.

## Results

Times are seconds; Q25–Q75 is the middle half of the six samples.

| Runner | Median | Minimum | Q25–Q75 | Standard / median |
| --- | ---: | ---: | ---: | ---: |
| Standard reactive | 5.705245 | 5.684705 | 5.697010–5.720092 | 1.000× |
| Wave runner, 1 worker | 8.821383 | 8.766264 | 8.800648–8.824901 | 0.647× |
| Wave runner, 2 workers | 9.565452 | 9.511125 | 9.550126–9.590793 | 0.596× |
| Wave runner, 4 workers | 9.288565 | 9.219593 | 9.283683–9.350493 | 0.614× |
| Wave runner, 6 workers | 9.244071 | 9.234180 | 9.240963–9.285772 | 0.617× |

| Runner | Round 1 median | Round 2 median | Median GC time | Allocated bytes per call* | Allocations* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 5.723826 | 5.701600 | 0.477790 | 2,164,172,992 | 49,312,242 |
| 1 worker | 8.794374 | 8.825436 | 0.665844 | 3,197,296,752 | 63,243,922 |
| 2 workers | 9.598730 | 9.563922 | 0.654721 | 3,236,176,880 | 65,655,818 |
| 4 workers | 9.285488 | 9.291642 | 0.640820 | 3,236,381,680 | 65,657,818 |
| 6 workers | 9.299088 | 9.240511 | 0.670598 | 3,236,586,480 | 65,659,818 |

*BenchmarkTools' minimum allocation estimates across the trials, not peak live
memory. Allocation totals can exceed peak RSS because memory is collected and
reused during an inference.

## Accuracy and interpretation

Relative residual is `norm(A * mean_x - b, Inf) / norm(b, Inf)`.
Relative mean error is measured against `sparse(A) \ b` in the infinity norm.

| Schedule | Relative residual | Relative mean error |
| --- | ---: | ---: |
| Standard | 1.193916e-7 | 6.892459e-7 |
| Wave, all tested worker counts | 6.379920e-5 | 1.099768e-4 |

Means and variances matched exactly across wave-runner worker counts and on
every untimed preflight/post-trial check. They are not claimed to match the
standard schedule at the same finite iteration count. These are fixed-iteration
timings, **not time-to-equal-accuracy results**.

With multiple workers, 2,431,964 jobs were dispatched to workers and 36 ran
serially over 200 waves. That is a count of scheduled jobs, not the fraction of
total runtime parallelized. Graph construction, snapshotting, reactive delivery
and equality-chain caching remain serial. The extra allocation volume and
slower single-worker wave runner are consistent with execution overhead; no
profile attributing the regression to one particular operation was collected.

The full 48,000-cell heat-grid state-space example remains unmeasured. Positive
results on dense multivariate Gaussian models do not establish scaling here.

## Raw samples and reproduction

Seconds, round 1 followed by round 2:

```text
standard: 5.695479875 5.759297208 5.723825709 5.708890958 5.701599958 5.684705042
1 worker: 8.766263625 8.794374000 8.823295083 8.881234875 8.825435792 8.819471583
2 workers: 9.598729958 9.611808708 9.545527125 9.563921792 9.566982083 9.511124583
4 workers: 9.285487792 9.219593458 9.476839125 9.291642167 9.370109417 9.283081000
6 workers: 9.245824125 9.867334000 9.299088333 9.242318708 9.234180333 9.240510625
```

Full [BenchmarkTools trials](/private/tmp/rxinfer-benchmarktools.NQeR2S/linear64-blas6.json)
and the compatible [environment](/private/tmp/rxinfer-benchmarktools.NQeR2S/Project.toml)
are retained locally in the temporary directory. The environment was seeded
from the working RxInfer Julia 1.11 Manifest, preserving dependency versions
while adding BenchmarkTools and its dependencies. Both RxInfer and ReactiveMP
resolve to the local modified checkouts. Repository manifests were not changed.

Run from the workspace root, while that temporary environment is retained:

```sh
RXINFER_BENCHMARK_SAMPLES=3 RXINFER_BENCHMARK_ROUNDS=2 julia --startup-file=no --project=/private/tmp/rxinfer-benchmarktools.NQeR2S --threads=6 --heap-size-hint=1536M RxInfer.jl/benchmarks/multicore_linear_system.jl 64 100 0.1
```

Use `RXINFER_BENCHMARK_OUTPUT` with a new filename to save another raw trial
file. The checked-in general benchmark Manifest targets Julia 1.12.5, so it
cannot be substituted unchanged into this Julia 1.11 comparison.

Verification: 41 harness checks passed; all four benchmark scripts passed small
model smoke checks, including posterior assertions and raw-trial saving. No
runner implementation changed during this benchmark-methodology revision.
