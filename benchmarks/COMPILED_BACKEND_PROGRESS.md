# Compiled backend implementation

Acceptance: broad converged-posterior compatibility, batch and streaming, explicit
unsupported-feature errors, and an actual 1000×1000 grid below 8 GiB peak RSS.
Capacity/correctness gate completion; multicore speedup must be measured, not promised.

## Status

- [x] Compact GraphPPL construction, metadata, and initial compatibility tests.
- [x] Non-Rocket rule interpreter and generic factor lowering.
- [x] Initial constraints, nonlinear/custom-layout adapters, scoring, callbacks and predictions, with explicit unsupported cases.
- [x] Streaming/autoupdates, pause/restart and failure poisoning/recovery via a fresh instance.
- [x] General typed payload storage with guarded boxed deoptimization; coarse deterministic parallel execution.
- [ ] Full model compatibility matrix.
- [x] Actual million-variable capacity and convergence gate (staged revision measurements documented below).
- [x] BenchmarkTools construction/inference/end-to-end accuracy-matched report.

The full acceptance gate has **not** passed. The existing default reactive
backend, MulticoreRunner, and unrelated worktree changes are preserved.

## Validated so far (Julia 1.11.9, six Julia threads)

- GraphPPL compact-storage plus core-engine tests: 60,331 assertions pass across
  75 test items (including 1,055 compact-storage assertions). Includes metadata
  presence/widening, ordered connectivity, shared labels, model expansion and partitions.
- GraphPPL variational-constraint engine, macro and integration tests: all 3,665
  assertions pass in the final reproducible environment.
- ReactiveMP execution/storage tests: 16,421 assertions pass. Includes dependency
  phases, failure poisoning, typed-payload type changes, annotation fallback and
  rejecting unavailable dependencies even when a requested marginal is initialized.
- Existing ReactiveMP message/marginal/variable/Mixture-switch tests: 2,150 assertions pass.
- Existing RxInfer inference and model-construction suites (excluding the new
  compiled file): 12,788 assertions pass. Includes predictions, sparse data,
  autoupdates, initialization and the earlier wave runner.
- Aqua checks pass for GraphPPL, ReactiveMP and RxInfer with the repositories'
  standard ambiguity/piracy exclusions. An unbound wave-job constructor was fixed
  by requiring its result type explicitly, as its existing caller already does.
  The focused wave-runner suite then passes all 87 assertions, including the new
  explicit-result constructor checks.
- RxInfer compact tests: all 150 assertions pass in a clean run in
  `benchmarks/compiled`, including the latest startup-dependency guard. Conjugate
  Gaussian/Beta/Gamma, structured HMMs, Probit self-message policies, Gaussian and
  Gamma mixtures, point-mass constraints, Mixture annotations, nonlinear Delta,
  missing-data predictions, streaming q/μ autoupdates, variable iterations,
  restart and failure cleanup. CVI checks use
  three seeds and explicit Monte Carlo tolerances, not trajectory equality.
- Actual RSLDS notebook fixture, horizon 12, 1,000 iterations: 20 mean/variance
  assertions pass across ten posterior families, relative errors around 1e-15.
  Both final free energies are 46.9114890115418 within floating-point roundoff.
  Rechecked successfully after the final startup guard (all 20 assertions pass).
  The 100-iteration fixture had not converged and did not pass posterior tolerance.
- Single-use VMP messages are fused with their marginal product before coloring:
  coordinate updates resolve a Gamma-mixture oscillation exposed by the initial
  message/product-level Jacobi schedule. Numerical rules and products are reused.
- Loopy scalar grids 4/8/16 match converged standard posterior means/variances.
- 500×500 (250,000 unknowns) actually converged in the earlier boxed prototype:
  160 sweeps, relative infinity-norm residual 9.9422e-7, 20 consecutive stable
  posterior sweeps, 3.7358 GiB peak RSS, 1.6999 GiB final live heap. Float64,
  leak 0.1, sinusoidal RHS, KeepLast, workers=6, BLAS=1.
- Latest 64×64 capacity check passes: 157 sweeps, residual 9.9798e-7,
  16 consecutive stable posterior sweeps, 0.9138 GiB peak RSS.
- Three completed million-variable attempts have failed the 8 GiB gate:
  8.6408 GiB and 8.5757 GiB before inference; then 7.9265 GiB at construction
  completion and 8.0142 GiB as inference started. All stopped with exit 2.
  None was a million-variable posterior result.
- Fourth 1000×1000 attempt **passes**: 1,000,000 unknowns, 158 sweeps,
  residual 9.8859e-7, 17 consecutive stable mean/variance sweeps,
  7.7708 GiB peak process RSS and 5.2138 GiB final live heap. Exit 0.
  Construction peak was 7.2879 GiB. Float64, leak 0.1, sinusoidal RHS, KeepLast,
  workers=6, BLAS=1, Julia heap hint=4G. Same Gaussian factors, edge-list input;
  no dense million-square matrix and no specialized linear-system solver.

Memory revisions now include shared edge labels, exact-sized CSR arrays with
32-bit offsets when possible, flat compiler edge bindings, dead initialization
cleanup, and collection between large compilation phases. RSS watchdogs and
Julia heap hints are not hard OS caps; the successful gate is based on measured
peak RSS over the completed process, not the hint or a size extrapolation.

Full GraphPPL regression testing is **not passed**: `Nested model structure`
segfaults both in this worktree and in unmodified registry v4.8.0 on Julia 1.11.9.
The remaining suite initially exposed a metadata return-type inference regression;
the fix now passes all 60,331 compact/core-engine assertions. `haskey` has a Bool
contract, and `setextra!` returns the node (previously the underlying dictionary),
keeping updates type-stable across storage backends. Existing in-repo callers
ignore that return value; extension callers relying on it must use `getextra(node)`.
Do not describe full default-backend regression coverage as complete.
Subsequent broader GraphPPL runs also exposed missing standalone test dependencies
in the benchmark environment (MacroTools, then BitSetTuples), not failed numerical
assertions. Those dependencies are now present, and the full affected
variational-constraint subset passes all 3,665 assertions. The other stopped
full-suite runs still must not be described as passes.

The machine exhibited heavy memory pressure during the successful capacity run,
and other validation processes ran during part of it. This was a capacity and
correctness measurement, not a timing benchmark. Later streaming/API/metadata
return-type fixes do not change this grid's numerical rules or state layout.
A later startup guard rejects unresolved dependencies instead of returning an
initialized prior; it adds no retained program fields. The million-variable probe
has not been repeated after that guard. Smaller-grid calibration is rerun for the
final performance report.

## Isolated sweep diagnostic (not end-to-end speedup)

BenchmarkTools, 64×64, evals=1, three samples per configuration per round, two
counterbalanced rounds, warm execution, BLAS=1 throughout, no concurrent Julia
workload. The fixture checks exact agreement between compiled configurations.

| Configuration | Round 1 median | Round 2 median |
| --- | ---: | ---: |
| Compiled boxed, 1 worker | 11.753 ms | 11.502 ms |
| Compiled typed, 1 worker | 8.570 ms | 8.566 ms |
| Compiled typed, 6 workers | 2.346 ms | 2.389 ms |

This shows 3.59–3.65× one-to-six-worker scaling for the isolated typed sweep.
It does not establish an end-to-end improvement over standard inference, or
linear scaling on other models. Threaded allocation counters varied between
samples/rounds; do not infer precise allocation reductions from this diagnostic.

The table predates the latest memory and coordinate-fusion changes. Reproduce
with `compiled_sweep_diagnostic.jl`; capacity checks use `compiled_grid_capacity.jl`.
`compiled_benchmark_tools.jl` adds equal-accuracy calibration and separate
construction/setup, inference and public-API end-to-end measurements. Its 8×8
correctness-only preflight passes (85 reference sweeps versus 168 compiled sweeps);
the final 64×64 timed comparison passes with retained raw trials and source/configuration
metadata under `compiled/results/grid64-d3cLOr`. Pooled medians: standard BLAS=6
4.654 s end-to-end / 4.296 s inference; compiled workers=6/BLAS=6 1.869 s /
0.937 s. That is 2.49× end-to-end and 4.58× inference speedup over the reference.
One-to-six-worker scaling at BLAS=1 is only 1.73× inference / 1.35× end-to-end;
compiled construction is 2.80× slower than the reference. No linear scaling claim.
The earlier run's temporary raw file was deleted at Julia exit and is superseded,
not pooled with the retained run. Accuracy checks run outside timing, and inference
fixtures start from fresh initialized state for every sample. See
[the complete report](COMPILED_BENCHMARKTOOLS.md) for phase boundaries, stack-depth
and BLAS settings, round medians and raw data links.

Use the relative-path environment and resolved manifest in `benchmarks/compiled`.
GraphPPL changes are isolated in the
`GraphPPL-compiled.jl` worktree on `codex/compiled-backend`, based on v4.8.0.
The user's older `GraphPPL.jl` branch was left unchanged.
