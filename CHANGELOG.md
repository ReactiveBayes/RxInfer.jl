# Changelog

All notable changes to RxInfer.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- The `callbacks` argument in the `infer` function now accepts any custom structure that implements `ReactiveMP.invoke_callback`, in addition to `NamedTuple` and `Dict`. The available callbacks list now also includes ReactiveMP-level callbacks such as `before_message_rule_call`, `after_message_rule_call`, `before_product_of_messages`, `after_product_of_messages`, `before_marginal_computation`, `after_marginal_computation`, and others.
- **Breaking:** `before_iteration` and `after_iteration` callbacks must now return `StopIteration()` to halt iterations instead of `true`. Any other return value (including `nothing`) continues iteration. The `StopEarlyIterationStrategy` has been updated accordingly.
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

[Unreleased]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.7.2...HEAD
[4.7.3]: https://github.com/ReactiveBayes/RxInfer.jl/compare/v4.7.2...HEAD
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
