# Plan: Refactor RxInfer addons to annotations

## Context

ReactiveMP.jl (on its `refactor-addons` branch, targeting `release-6`) has renamed "addons" to "annotations" with a new architecture:
- `AbstractAddon` -> `AbstractAnnotations`
- `getaddons()` -> `getannotations()`
- `AddonLogScale` -> `LogScaleAnnotations`
- `AddonMemory` -> `InputArgumentsAnnotations`
- `Marginal{D,A}` -> `Marginal{D}` (annotations are now always an `AnnotationDict`, not a type parameter)
- `FactorNodeActivationOptions.addons` -> `.annotations`
- `MessageProductContext` gains an `annotations` field (replaces addons concept)

RxInfer needs to match these API changes so it works with the new ReactiveMP release.

## Files to modify

### 1. `src/model/model.jl` (line 12) — DONE
- Change `import ReactiveMP: getaddons, AbstractFactorNode` to `import ReactiveMP: getannotations, AbstractFactorNode`

### 2. `src/model/plugins/reactivemp_inference.jl` — DONE
- Renamed struct field `addons::A` -> `annotations::A` and updated all constructors
- Renamed `setaddons` -> `setannotations`; all internal `options.addons` -> `options.annotations`
- `convert` method: `:addons` -> `:annotations` in `available_options`, local variable, and `haskey` checks
- Updated import: `getaddons` -> `getannotations`
- Replaced all `ReactiveMP.getaddons` dispatches with `ReactiveMP.getannotations`
- Replaced `ReactiveMP.AbstractAddon` with `ReactiveMP.AbstractAnnotations` in tuple-wrapping dispatch
- `activate_rmp_factornode!`: passes `annotations` instead of `addons` to `FactorNodeActivationOptions`
- `activate_rmp_variable!`: added `annotations = getannotations(getoptions(plugin))` to both `MessageProductContext` constructors

### 3. `src/inference/postprocess.jl` — DONE
- Removed `DefaultPostprocess` entirely.
- `UnpackMarginalPostprocess` always unpacks the `Marginal` wrapper.
- `NoopPostprocess` always preserves the result as-is.
- The decision of which strategy to use moved to `batch.jl`/`streaming.jl` (based on whether `annotations` is `nothing`).
- Updated all docstrings with detailed explanations and cross-references.

### 4. `src/inference/batch.jl` — DONE
- Renamed parameter `addons = nothing` -> `annotations = nothing`
- Renamed `postprocess = DefaultPostprocess()` -> `postprocess = nothing`
- Updated warning/override logic: `getaddons`/`setaddons` -> `getannotations`/`setannotations`, warning text updated
- Added automatic postprocess selection: `UnpackMarginalPostprocess()` when no annotations, `NoopPostprocess()` otherwise

### 5. `src/inference/streaming.jl` — DONE
- Same changes as batch.jl: renamed `addons` -> `annotations`, `postprocess` default to `nothing`
- Updated warning/override logic with new function names and text
- Added automatic postprocess selection logic

### 6. `src/inference/inference.jl` — DONE
- Renamed `addons = nothing` -> `annotations = nothing` in docstring and signature
- Renamed `postprocess = DefaultPostprocess()` -> `postprocess = nothing` in docstring and signature
- Updated docstring for `annotations` with detailed explanation, examples, and pointer to ReactiveMP docs
- Updated docstring for `postprocess` with explanation of defaulting behavior
- Passed `annotations` instead of `addons` to `batch_inference` and `streaming_inference` calls
- Updated session logging: `ctx[:addons]` -> `ctx[:annotations]`

### 7. Tests — DONE

**`test/model/plugins/reactivemp_inference_tests.jl`:**
- Replaced `AbstractAddon` with `AbstractAnnotations`
- Renamed `MyAddons`/`MyAnotherAddons` -> `MyAnnotations`/`MyAnotherAnnotations`
- Replaced all `getaddons` -> `getannotations`, `setaddons` -> `setannotations`

**`test/inference/postprocessing_tests.jl`:**
- Removed `DefaultPostprocess` test block
- Updated `Marginal` constructors: 3-arg for no-annotations case, `AnnotationDict()` + `annotate!` for with-annotations case
- Updated comments

**`test/inference/inference_tests.jl` (lines 704-720):**
- Replaced `addons = AddonLogScale()` with `annotations = LogScaleAnnotations()`
- Updated warning message match text from "addons" to "annotations"
- Updated `options = (addons = ...)` to `options = (annotations = ...)`

**`test/models/mixtures/mixture_tests.jl` (lines 48-73):**
- Replaced `addons = AddonLogScale()` with `annotations = LogScaleAnnotations()`

### 8. Documentation — DONE

**`docs/src/manuals/inference/postprocess.md`:** DONE
- Rewrote page: added "Default behavior" section explaining automatic strategy selection
- Removed `DefaultPostprocess` from docs block
- Improved prose throughout

**`docs/src/manuals/debugging.md`:** DONE
- Updated `AddonMemory` section to use `InputArgumentsAnnotations`
- Replaced `addons = (AddonMemory(),)` with `annotations = (InputArgumentsAnnotations(),)`
- Replaced `getaddons` with `getannotations`
- Updated section headings, prose, and the legacy warning note

**`docs/src/manuals/inference/overview.md`:** DONE (no changes needed — file contains no addon references)

## Key design decisions

1. **Postprocessing logic (REVISED)**: `DefaultPostprocess` has been removed. Instead, `batch_inference` and `streaming_inference` choose the default postprocess strategy based on the `annotations` argument: `UnpackMarginalPostprocess()` when `annotations` is `nothing`, `NoopPostprocess()` otherwise. If the user explicitly passes `postprocess`, their choice is always respected.

2. **MessageProductContext annotations**: RxInfer currently doesn't pass annotations to `MessageProductContext`. With the new ReactiveMP, annotation processors need to be passed there so `post_product_annotations!` works. Add `annotations = getannotations(getoptions(plugin))` to both message and marginal `MessageProductContext` constructors in `activate_rmp_variable!`.

3. **Tuple wrapping**: The old code wrapped single `AbstractAddon` instances into tuples. The new ReactiveMP uses annotation processors differently (they're passed to `MessageProductContext` and `FactorNodeActivationOptions`). Need to check if tuple wrapping is still needed or if the new API expects individual processors or a collection.

## Status

All refactoring is complete. User will run tests manually for verification.

## Breaking changes (for CHANGELOG)

- **`addons` keyword renamed to `annotations`** in `infer()`, `batch_inference()`, and `streaming_inference()`. Users must update `infer(..., addons = AddonLogScale())` to `infer(..., annotations = LogScaleAnnotations())`.
- **`DefaultPostprocess` removed**. The `postprocess` keyword now defaults to `nothing`, and the strategy is chosen automatically based on whether `annotations` is set. Users who explicitly passed `postprocess = DefaultPostprocess()` should remove it or pass `nothing`.
- **`AddonLogScale` renamed to `LogScaleAnnotations`** (from ReactiveMP).
- **`AddonMemory` renamed to `InputArgumentsAnnotations`** (from ReactiveMP).
- **`getaddons` / `setaddons` renamed to `getannotations` / `setannotations`** on `ReactiveMPInferenceOptions`.
- **`Marginal` constructor changed**: `Marginal(data, is_point, is_clamped, addons)` is now `Marginal(data, is_point, is_clamped)` (3 args) or `Marginal(data, is_point, is_clamped, annotation_dict)` with an `AnnotationDict`. The `Marginal` type no longer has a type parameter for addons (`Marginal{D}` instead of `Marginal{D,A}`).
- **`options = (addons = ...,)` renamed to `options = (annotations = ...,)`** in NamedTuple-based option passing.
