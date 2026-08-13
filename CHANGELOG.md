# Changelog

All notable changes to RxInfer.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Fixed
- `SampleListFormConstraint` with the `AutoProposal` strategy now raises an actionable error (pointing the user at `LeftProposal`/`RightProposal`) when neither operand of the product is a low-priority proposal candidate, instead of falling through to a cryptic `MethodError`. Added a regression test. ([#681](https://github.com/ReactiveBayes/RxInfer.jl/issues/681))

## [5.5.1] - 2026-08-12

### Added
- Control over whether the **model source code** is included when sharing session data. Session sharing has always included the model source (`GraphPPL.getsource`) together with the `constraints`/`meta` source blocks; this is now opt-out. A new compile-time preference (default `true` = share) can be toggled with `RxInfer.enable_source_code_sharing!()` / `RxInfer.disable_source_code_sharing!()`, and `share_session_data` gains a `share_source_code` keyword (`nothing` follows the preference, `true`/`false` overrides it). When disabled, the `model`/`constraints`/`meta` fields are replaced with a `"<redacted>"` marker in the shared payload; the local session keeps the full context. The telemetry manual now documents exactly what is transmitted. ([#682](https://github.com/ReactiveBayes/RxInfer.jl/issues/682))

### Changed
- Made Julia code formatting deterministic across CI and local runs. The `scripts/` formatter environment previously re-resolved `JuliaFormatter` to the newest version on every `make format`/`make lint` (`Pkg.update()` in `scripts_init`) and was neither compat-bounded nor run on a pinned Julia version in CI, so the same code could be formatted differently depending only on when/where the formatter ran — producing spurious "🤖 Auto-format Julia code" PRs. `scripts_init` now only instantiates the pinned `scripts/Manifest.toml` (deliberate bumps moved to a new `make scripts_update`), `scripts/Project.toml` pins `JuliaFormatter = "~2.12"`, and the `format-check` CI job pins Julia to `1.12` (JuliaFormatter's output can shift with the Julia minor version via `JuliaSyntax`). Also refreshed several GitHub Actions to their Node 24 releases (`upload-artifact` v6, `codecov-action` v5, `checkout` v6, `julia-actions/cache` v3) to clear Node 20 deprecation annotations. Includes a one-time reformat of the repository under the pinned formatter.

### Fixed
- `RxInfer.share_session_data()` no longer throws `UndefVarError: data not defined` when re-uploading an already-registered document (the PATCH branch of `__add_document`). Repeat manual sharing and automatic session sharing (which hits this path on every inference call after the first) now update the existing Firestore document correctly. Added a regression test that exercises the update branch without network access. ([#679](https://github.com/ReactiveBayes/RxInfer.jl/issues/679))
- Guarded the module-global `id_name_mapping` telemetry `Dict` against concurrent access. It was read and written directly from background telemetry tasks (`log_using_rxinfer` and automatic session sharing both dispatch via `Base.Threads.@spawn`), a latent data race that could lose updates or corrupt the `Dict` during a hash resize under multi-threading. All access now goes through locked `__get_document_name` / `__set_document_name!` accessors backed by a `ReentrantLock`. ([#683](https://github.com/ReactiveBayes/RxInfer.jl/issues/683))

## [5.5.0] - 2026-06-18

### Added
- Support for inference over **partially-referenced (sparse) conditioned data tensors**, in both batch (`infer` with `data`) and streaming (`infer` with `datastream`) modes. When a model conditions on a data array but only references some of its indices in `~` statements (e.g. masked / missing observations, or a sub-model that touches only the observed entries), GraphPPL materializes data-variable labels only at those indices, leaving the rest of the bounding box as `#undef` holes. Result collection (`getvardict`/`getvarref`), the random/data/anonymous predicates, and the per-iteration / per-tick data feed now iterate only the *assigned* entries, and the supplied dense data is fed to the materialized variables **by index** (unreferenced entries are ignored). Dense data tensors are unaffected. See the new [Partially-referenced (sparse) data](https://docs.rxinfer.com/stable/manuals/inference/partial-data/) manual page.
- Support for conditioning on **data with non-standard (offset) indexing**, e.g. an `OffsetArray` whose axes start at `0` (or a negative index). Such data is now presented to the model with standard 1-based axes (values and order preserved), so models index it the usual way (`1:n`, `eachindex`, `axes`, …) and partial/sparse referencing works too. Standard 1-based arrays are returned unchanged with no copy. Indexing the model with a literal offset index (e.g. `y[0]`) remains unsupported.

## [5.4.0] - 2026-06-09

### Added
- Support for inline specification of submodel initializations. Instead of a for-loop at the top level to specify initialization for variables in submodels, you can now specify the initialization inline with the submodel calls.
- A small "or ask DeepWiki" link below the "Search with Gemini" widget in the documentation sidebar, pointing to the same [DeepWiki page](https://deepwiki.com/ReactiveBayes/RxInfer.jl) as the README badge. ([#670](https://github.com/ReactiveBayes/RxInfer.jl/pull/670))

### Changed
- Comprehensive documentation and docstring polishing pass across all documentation pages (`docs/src/**`, `README.md`) and source docstrings (`src/**`, `ext/**`). Fixes typos and doubled words, grammar and clarity, broken/missing Documenter cross-references, and stale content — including repointing the `index.md` `@contents` block to pages that actually exist, replacing the renamed `rxinference` function name with `infer` in inference/streaming docstrings and error messages, correcting the `objective_diagnostics` keyword to `free_energy_diagnostics`, fixing `result.posterior` to `result.posteriors` in the getting-started guide, and correcting several docstring constructor signatures to match the code (`IndividualAutoUpdateSpecification`, `AutoUpdateMapping`, `with_session`, `SampleListFormConstraint`). No executable code, function signatures, or doctest outputs were changed. ([#671](https://github.com/ReactiveBayes/RxInfer.jl/pull/671))

## [5.3.4] - 2026-06-03

### Fixed
- Restored the "Search with Gemini" widget in the documentation. The Google Cloud project backing the previous Vertex AI Search `configId` no longer existed, so the widget silently stopped working. `docs/src/assets/chat.js` now points at a newly created Vertex AI Search (AI Applications) app, documents the full setup in a header comment (no API key lives in the repo — the data store, public access, and domain allowlist are configured in the GCP console; website data stores require an Enterprise-edition search app), and logs a `console.warn` instead of failing silently when the widget or the Google SDK fails to load. The same `configId` is shared with the RxInferExamples.jl documentation. ([#668](https://github.com/ReactiveBayes/RxInfer.jl/pull/668))

## [5.3.3] - 2026-06-01

### Changed
- Updated `[compat]` entry for GraphPPL to `"4.7.0"` (allowing any version in the `[4.7.0, 5.0.0)` range), replacing the previous tilde-pinned `"~4.6.0"`.

## [5.3.2] - 2026-05-12

### Added
- `TensorBoardLoggerExt` now emits a `Summary` text tag alongside `EventCounts` with a single-snapshot timing rollup of the inference run: `model_build` (wall-clock between `BeforeModelCreationEvent` and `AfterModelCreationEvent`), `inference` (between `BeforeInferenceEvent` and `AfterInferenceEvent`), `total_wall` (first-to-last traced event), and per-iteration aggregates (`n_iterations`, `iter_total`, `iter_mean`, `iter_min`, `iter_max`). Lines are skipped silently when the corresponding measurement is missing — runs that bypass model creation or have no variational iterations still produce a useful Summary instead of an empty or misleading-zero table. The existing `iteration_time_ms` per-iteration scalar series is unchanged.
- `Base.show` methods for every Tier A callback event (`Before/AfterModelCreationEvent`, `Before/AfterInferenceEvent`, `Before/AfterIterationEvent`, `Before/AfterDataUpdateEvent`, `OnMarginalUpdateEvent`, `Before/AfterAutostartEvent`), so trace breadcrumbs in TBLogger's Text tab no longer dump raw struct contents ([#638](https://github.com/ReactiveBayes/RxInfer.jl/issues/638)). The methods honor the `IOContext` `:compact` flag: trace loggers pass `:compact => true` to get the short `EventName(model=Type, span=ab12…)` form, while REPL/Pluto/Jupyter sees the full form with the canonical struct constructor name and full UUID span id. The `_show_span` helper omits the field entirely when the span id is `nothing` (callbacks disabled). Pairs with the matching ReactiveMP-side work for `MessageMapping`, `MessageProductContext`, `AnnotationDict`, `FormConstraintCheck*`, and the Tier B events.

### Changed
- `TensorBoardLoggerExt` per-event text breadcrumbs now render via `sprint(show, ev; context = :compact => true)` (a small `_compact_repr` helper) instead of `repr(ev)`. After the event `Base.show` rework, `repr` returns the full interactive form (actual messages, full UUID); the `:compact => true` context yields the short trace-friendly form one log line wide. The previous bespoke `"k1: v1 | k2: v2 | …"` strings (and the now-removed `_format_fields` helper, ~110 LOC) are replaced by direct delegation to the event's own `show` method. ([#638](https://github.com/ReactiveBayes/RxInfer.jl/issues/638))
- Refactored `test/ext/TensorBoardLoggerExt/tensorboardlogger_tests.jl` for efficiency and maintainability. The 14 per-distribution scalar-dispatch tests (Poisson, Geometric, NegativeBinomial, Binomial, Exponential, VonMises, Weibull, LogNormal, Erlang, Laplace, Pareto, Rayleigh, Chisq, Uniform-fallback) collapse into a single table-driven `@testitem` with one `@testset` per row, and the three Summary-writer subtests collapse into a single `@testitem` with three `@testset`s. Shared `@model` / `@constraints` / `@initialization` boilerplate for the IID Normal, coin-toss, and IID InverseGamma fixtures moves into `helpers.jl` as `iid_normal_inference`, `coin_toss_inference`, and `iid_invgamma_inference` factories, plus a `with_dispatch_logger` helper that snapshots `(tags, steps)` after closing the TBLogger so the Windows EBUSY retry in `with_safe_tempdir` still applies. Net: ~30 → 17 testitems; ~1490 → ~480 lines; per-concern parallelism and failure granularity preserved via `@testset`. No behavioural change to extension code.

## [5.3.1] - 2026-05-05

- Relax `[compat]` entry for ReactiveMP.jl dependency, allowing any version in the `[6.0.0, 7.0.0)` range. Previously was too strict, allowing only versions `[6.0.0, 6.1.0)`.

## [5.3.0] - 2026-05-04

- Added Perfetto trace viewer support: `perfetto_view(trace)` converts a `RxInferTraceCallbacks` trace to Perfetto JSON, and `perfetto_open(trace)` opens it directly in the Perfetto UI in the browser.

## [5.2.1] - 2026-04-27

### Added
- `TensorBoardLoggerExt` posterior scalar logging now covers more univariate families. In addition to `Normal` (mean/precision) and `Gamma` (shape/rate), it now emits parameterisation-aware tags for `Beta` (alpha/beta/mean), `Bernoulli` (succprob), `Binomial` (ntrials/succprob), `InverseGamma` / `GammaInverse` (shape/scale), `Poisson` (rate), `Geometric` (succprob), `NegativeBinomial` (r/succprob), `Exponential` (rate), `VonMises` (location/concentration), `Weibull` (shape/scale), `LogNormal` (meanlog/stdlog), and `Erlang` (shape/scale), `Laplace` (location/scale), `Pareto` (shape/scale), `Rayleigh` (scale), and `Chisq` (dof). Any remaining `UnivariateDistribution` falls back to generic `mean` and `var` tags so unknown posteriors still produce visible convergence traces.
- `RxInfer.convert_to_tensorboard` accepts a new `log_posteriors` keyword that filters which marginals reach the `posteriors/*` tags. Pass `false` to suppress every posterior tag (scalars and histograms), `true` (default) to keep current behaviour, or a `Vector{String}` / `Vector{Symbol}` allow-list (e.g. `["μ"]` or `[:μ]`) to log only the named variables. Iteration timing, event counts, and event-text breadcrumbs are unaffected by the filter.

## [5.2.0] - 2026-04-24

### Added 
- The `trace = ...` keyword argument now accepts a tuple of symbols. In this case, 
  only the events, whose names are present in the tuple will be traced.

## [5.1.0] - 2026-04-23

### Added
- Added `TensorBoardLoggerExt` extension: when `TensorBoardLogger.jl` is loaded, `RxInfer.convert_to_tensorboard(trace)` exports an inference trace to TensorBoard event log files. Capabilities:
  - **Iteration timing** — wall-clock duration of each variational iteration logged as `iteration_time_ms`.
  - **Posterior scalars** — per-iteration mean/precision for `Normal` and shape/rate for `Gamma` marginals under `posteriors/<variable>/`.
  - **Posterior distributions** — per-iteration `HistogramSummary` (ridgeline + percentile bands in TensorBoard) via `log_distributions = true` and configurable `n_samples`.
  - **Event text breadcrumbs** — full per-event narrative (`Events`, `before_iteration`, `after_iteration`, etc.) gated behind `log_text_events = true` (off by default). `EventCounts` is always emitted as a compact run summary.
  - Logs are written to `tensorboard_logs/` in the current working directory by default; a custom path can be supplied via `output_file`.


## [5.0.0] - 2026-04-17

- **Breaking:** Addons have been renamed to annotations to match the new ReactiveMP API. This affects the `infer` function and related types:
  - The `addons` keyword argument in `infer()`, `batch_inference()`, and `streaming_inference()` has been renamed to `annotations`. Update `infer(..., addons = AddonLogScale())` to `infer(..., annotations = LogScaleAnnotations())`.
  - In NamedTuple-based options, `options = (addons = ...,)` is now `options = (annotations = ...,)`.
  - `AddonLogScale` has been renamed to `LogScaleAnnotations` (from ReactiveMP).
  - `AddonMemory` has been renamed to `InputArgumentsAnnotations` (from ReactiveMP).
  - `getaddons` / `setaddons` on `ReactiveMPInferenceOptions` have been renamed to `getannotations` / `setannotations`.
  - The `Marginal` constructor changed: `Marginal(data, is_point, is_clamped, addons)` is now `Marginal(data, is_point, is_clamped)` (3-arg) or `Marginal(data, is_point, is_clamped, annotation_dict)` with a `ReactiveMP.AnnotationDict`. The `Marginal` type no longer has a type parameter for addons (`Marginal{D}` instead of `Marginal{D, A}`).
  - See the ReactiveMP documentation for the new annotation processor API and how to implement custom annotations.
- **Breaking:** `DefaultPostprocess` has been removed. The `postprocess` keyword in `infer()` now defaults to `nothing`, and the strategy is selected automatically based on the `annotations` keyword: `UnpackMarginalPostprocess()` when `annotations` is `nothing` (the default), and `NoopPostprocess()` when annotations are enabled. If you previously passed `postprocess = DefaultPostprocess()` explicitly, simply remove it. Custom postprocessing strategies passed via `postprocess = ...` continue to work unchanged.
- **Breaking:** The callback system has been refactored to use event structs instead of dispatch with positional arguments. All callback events are now concrete structs subtyping `ReactiveMP.Event{E}` with named fields. Callbacks receive a single event object instead of positional arguments.
  - **NamedTuple/Dict callbacks**: Functions now receive a single event object instead of positional args. E.g. `(model, iteration) -> ...` becomes `(event) -> println(event.model, event.iteration)`.
  - **Custom callback structs**: The `callbacks` field of `infer` function now accepts custom structs that implement `ReactiveMP.handle_event(::MyCustomCallbacksHandler, event::SomeEvent)`.
  - **ReactiveMP events**: It is possible now to add callbacks to the events happening in ReactiveMP inference engine. See the documentation of ReactiveMP for the available events.
  - **`StopEarlyIterationStrategy`**: Now receives an `AfterIterationEvent` instead of `(model, iteration)`.
  - New RxInfer-level event types: `BeforeModelCreationEvent`, `AfterModelCreationEvent`, `BeforeInferenceEvent`, `AfterInferenceEvent`, `BeforeIterationEvent`, `AfterIterationEvent`, `BeforeDataUpdateEvent`, `AfterDataUpdateEvent`, `OnMarginalUpdateEvent`, `BeforeAutostartEvent`, `AfterAutostartEvent`.
  - Migration is straightforward: replace positional arguments with named field access on the event object. See the Callbacks section in the documentation for details.
- The `callbacks` argument in the `infer` function now accepts any custom structure that implements `ReactiveMP.handle_event`, in addition to `NamedTuple` and `Dict`. The available callbacks list now also includes ReactiveMP-level callbacks such as `before_message_rule_call`, `after_message_rule_call`, `before_product_of_messages`, `after_product_of_messages`, `before_marginal_computation`, `after_marginal_computation`, and others.
- **Breaking:** `before_iteration` and `after_iteration` callbacks now use a mutable `stop_iteration::Bool` field on the event (default `false`). Set `event.stop_iteration = true` from a callback to halt iterations early instead of returning `true`. The `StopEarlyIterationStrategy` has been updated accordingly.
- **Breaking:** Pipeline stages and the per-node `scheduler` argument have been replaced by the unified `ReactiveMP.AbstractStreamPostprocessor` abstraction (propagated from ReactiveMP v6).
  - The `where { pipeline = ... }` node clause has been removed together with `AbstractPipelineStage`, `LoggerPipelineStage`, `AsyncPipelineStage`, `ScheduleOnPipelineStage`, `DiscontinuePipelineStage`, `EmptyPipelineStage`, `CompositePipelineStage`, `apply_pipeline_stage`, and `schedule_updates`.
  - The `scheduler` option under `infer(..., options = ...)` has been renamed to `stream_postprocessors` and now expects a `ReactiveMP.AbstractStreamPostprocessor` (or `nothing`). To reproduce the old `schedule_on(scheduler)` behaviour, wrap the scheduler in a `ReactiveMP.ScheduleOnStreamPostprocessor`. The default is now `nothing` instead of `AsapScheduler()`.
  - `LoggerPipelineStage` has no direct replacement — use the `before_message_rule_call` / `after_message_rule_call` callbacks instead.
  - `options = (limit_stack_depth = N,)` continues to work unchanged; internally it is now expanded to `ReactiveMP.ScheduleOnStreamPostprocessor(RxInfer.LimitStackScheduler(N))`.
- The `infer` function now accepts a `benchmark = true` keyword argument that automatically merges `RxInferBenchmarkCallbacks` with user-provided callbacks. Benchmark results are accessible via `result.model.metadata[:benchmark]`.
- The `infer` function now accepts a `trace = true` keyword argument that automatically merges `RxInferTraceCallbacks` with user-provided callbacks. All callback events are recorded as `TracedEvent` and accessible via `result.model.metadata[:trace]`.
- Added `TensorBoardLoggerExt` extension: when `TensorBoardLogger.jl` is loaded, `RxInfer.convert_to_tensorboard(trace)` exports inference trace events to TensorBoard event log files. The output directory can be specified via the `output_file` keyword argument. Tests are located in `test/ext/TensorBoardLoggerExt/tensorboardlogger_tests.jl`.
- Tests are now running with `TestItemRunner` instead of `ReTestItems`
- `infer` function got a new keyword argument `disable_inference_error_hint` that disables the inference error hint if set to `true`
- The inference error hint now can be forced to throw an error with `THROW_ON_INFERENCE_ERROR_HINT` environment variable. This is done primarily to catch errors on CI when a test prints this unintentionally (which also confuses our developers in [$606](https://github.com/ReactiveBayes/RxInfer.jl/issues/606)). All tests or documentation examples now need to use the `disable_inference_error_hint` if the error is intentional.

## [4.7.3] - 2026-03-13

### Documentation
- Added docstring for the `@initialization` macro ([603](https://github.com/ReactiveBayes/RxInfer.jl/pull/603))
- Added `CONTRIBUTING.md`
- Started using `CHANGELOG.md` (first few entries were auto-generated with LLM based on tag log)

## [4.7.2] - 2026-03-12

### Changed
- Relaxed gamma mixture tolerance on free energy (flaky test)

### Documentation
- Added cross-reference to RxInfer examples

## [4.7.1] - 2026-03-04

### Fixed
- Corrected Firestore endpoint URL construction and payload format in telemetry

### Documentation
- Updated debugging documentation with "Using callbacks in the infer function" section
- Fixed documentation build configuration

## [4.7.0] - 2026-02-16

### Added
- `StopEarlyIterationStrategy` callback for early stopping support (#595)

### Changed
- Added versioned Manifest to gitignore

## [4.6.7] - 2026-01-23

### Fixed
- Resolved `MixedArguments` bug in initialization macro (#585)

### Documentation
- Updated README.md

## [4.6.6] - 2025-12-18

### Changed
- Added JSON 1.0 compatibility (#589)

### Fixed
- Removed wrong source section from Project.toml

## [4.6.5] - 2025-11-24

### Added
- New documentation section "What is a rule" (#582)

### Changed
- Removed vibe coded fields from issue template (#581)

## [4.6.4] - 2025-11-20

### Added
- Issue templates (#568)
- Additional full pipeline tests
- Implemented backend-aware node alias conversion for initialization macro (#525)

### Changed
- Reimplemented kwargs init macro (#571)
- Updated benchmarks (#576)
- Made tags consistent (#577)

### Fixed
- Fixed documentation build (#580)
- Select tests with `make test test_args...` and disabled Aqua with `RUN_AQUA=true make test ...` (#563)

## [4.6.3] - 2025-11-04

### Added
- LiveServer as a dependency (#556)

### Changed
- Switched from CpuId to Hwloc for parallel tests (#544)
- Differentiated RxInfer.jl documentation from RxInferExamples.jl (#511)

### Fixed
- Added special case for `PointMassFormConstraint` for `Categorical` (#546)
- Fixed spelling and grammar mistakes in documentation (#552)

## [4.6.2] - 2025-10-21

### Fixed
- Fixed multi-agent path planning (use unscented instead)

### Tests
- Updated tests (#541)

## [4.6.1] - 2025-10-21

### Added
- Discord badge to README (#517)

### Changed
- Changed dispatch version of default_parametrization for GammaShapeScale (#526)

### Fixed
- Fixed test in PR526

## [4.6.0] - 2025-09-23

### Added
- New documentation section about static and streamlined inferences (#503)
- Test model for non-linear node (univariate -> multivariate) (#505)
- Support for streaming inference without auto-updates (#510)

### Changed
- Updated dependencies (#514)

---

[Unreleased]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.5.0...HEAD
[5.5.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.4.0...v5.5.0
[5.4.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.3.4...v5.4.0
[5.3.4]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.3.3...v5.3.4
[5.3.3]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.3.2...v5.3.3
[5.3.2]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.3.1...v5.3.2
[5.3.1]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.3.0...v5.3.1
[5.3.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.2.1...v5.3.0
[5.2.1]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.2.0...v5.2.1
[5.2.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.1.0...v5.2.0
[5.1.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.1.0...v5.0.0
[5.0.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v5.0.0...v4.7.3
[4.7.3]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.7.3...v4.7.2
[4.7.2]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.7.1...v4.7.2
[4.7.1]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.7.0...v4.7.1
[4.7.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.7...v4.7.0
[4.6.7]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.6...v4.6.7
[4.6.6]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.5...v4.6.6
[4.6.5]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.4...v4.6.5
[4.6.4]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.3...v4.6.4
[4.6.3]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.2...v4.6.3
[4.6.2]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.1...v4.6.2
[4.6.1]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.6.0...v4.6.1
[4.6.0]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.5.2...v4.6.0
