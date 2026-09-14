# Experimental compiled backend

For a short, self-contained demonstration with a nine-line model, see
[the Gaussian smoothing demo](../../../RxInferExamples.jl/compiled_runner_demo.md).
It compares complete inference calls at matched accuracy with BenchmarkTools.

This environment uses the three modified local packages, not registry releases.
Keep `RxInfer.jl`, `ReactiveMP.jl`, `GraphPPL-compiled.jl` and `RxInferExamples.jl`
as siblings. The GraphPPL worktree is based on v4.8.0 and includes the compact
storage changes; an unmodified v4.8.0 is insufficient. Julia 1.11 or newer is
needed for this environment's relative `[sources]` entries.

From `RxInfer.jl`:

```sh
julia --project=benchmarks/compiled -e 'using Pkg; Pkg.instantiate()'
LOG_USING_RXINFER=false julia --project=benchmarks/compiled --threads=6 -e 'using RxInfer, TestItemRunner; TestItemRunner.run_tests(pkgdir(RxInfer); filter=ti -> endswith(ti.filename, "compiled_tests.jl"))'
julia --project=benchmarks/compiled --threads=6 benchmarks/compiled_rslds_check.jl 12 1000
julia --project=benchmarks/compiled --threads=6 benchmarks/compiled_sweep_diagnostic.jl 64
julia --project=benchmarks/compiled --threads=6 benchmarks/compiled_benchmark_tools.jl 64
julia --project=benchmarks/compiled --threads=6 --heap-size-hint=4G benchmarks/compiled_grid_capacity.jl 1000 6
```

Run timing benchmarks alone, without other Julia workloads. The sweep diagnostic
uses BenchmarkTools (`evals=1`, three samples, two counterbalanced rounds), but
does **not** measure construction, result materialization or reference-backend
speedup. The separate `compiled_benchmark_tools.jl` harness calibrates sweep counts
to the same residual/posterior stability criterion, checks posterior agreement,
and measures construction/setup, fresh-state inference and public-API end-to-end
costs. Construction/setup includes attaching and detaching the posterior sink;
inference includes KeepLast materialization. These isolated phases are not claimed
to sum exactly to end-to-end API cost. Configurations include standard BLAS=6,
compiled workers=1/BLAS=1, and compiled workers=6 with BLAS=1 and BLAS=6.
The harness prints a `RAW_TRIALS` path and saves every BenchmarkTools sample there.
The output directory survives Julia's exit. The adjacent `metadata.json` records
calibration accuracy, package-source hashes, environment/harness hashes and run
configuration so measurements remain attributable to the tested implementation.

Capacity checks measure peak RSS, not performance. A heap hint and the
RSS watchdog are best-effort controls, not hard OS-enforced memory limits.

Opt in through `options=(runner=CompiledRunner(workers=6),)`. This backend does
not construct reactive factor/message networks. Existing numerical rules are
executed using indexed dependencies. Stateful rules, numerical callbacks and
unknown extension state are conservatively serialized. Unadapted imperative
node layouts and stream postprocessors produce explicit errors, never a silent
reactive fallback. Public streaming inputs/outputs still use Rocket subjects.

See `../COMPILED_BACKEND_PROGRESS.md` for actual validation results and remaining
acceptance work. This is still experimental, not a claim of all-model coverage.

Additional regression commands (run separately from timing benchmarks):

```sh
LOG_USING_RXINFER=false julia --project=benchmarks/compiled --threads=6 -e 'using ReactiveMP, TestItemRunner; TestItemRunner.run_tests(pkgdir(ReactiveMP); filter=ti -> occursin("/compiled/", ti.filename))'
LOG_USING_RXINFER=false julia --project=benchmarks/compiled --threads=6 -e 'using RxInfer, TestItemRunner; TestItemRunner.run_tests(pkgdir(RxInfer); filter=ti -> occursin("/inference/", ti.filename) || occursin("/model/", ti.filename))'
LOG_USING_RXINFER=false julia --project=benchmarks/compiled --threads=6 -e 'using Aqua, GraphPPL, ReactiveMP, RxInfer; for package in (GraphPPL, ReactiveMP, RxInfer); Aqua.test_all(package; ambiguities=false, piracies=false, deps_compat=(; check_extras=false, check_weakdeps=true)); end'
julia --project=benchmarks/compiled --threads=2 -e 'using GraphPPL, TestItemRunner; TestItemRunner.run_tests(pkgdir(GraphPPL); failfast=true, filter=ti -> occursin("/plugins/variational_constraints/", ti.filename))'
```

The broader default-backend tests can take substantially longer than the compact
fixtures. Full GraphPPL testing is not claimed complete: the independently
reproduced upstream `Nested model structure` crash is documented in the progress
record, and GraphViz integration is outside this validation environment.
