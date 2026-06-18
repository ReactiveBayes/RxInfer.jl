# [Partially-referenced (sparse) data](@id partial-data)

RxInfer materializes the nodes of a conditioned data array **lazily**: a data variable is
created only for the indices that your model actually references inside a `~` statement.
This means you can condition on a *full, dense* data array but only wire up the entries that
are relevant for a particular run — the unreferenced entries are simply ignored.

This is useful whenever the structure of the graph depends on the data that happens to be
available:

- **Masked / missing observations** — at each time step (or for each sensor) only some
  measurements are present, so only those should enter the likelihood.
- **Data-driven sub-graphs** — a sub-model that branches on a mask and only connects the
  observed indices, leaving the rest of a conditioned tensor untouched.
- **Ragged observation sets** — a different number of observations per group, padded into a
  rectangular array for convenience but only partially used.

## Example

The model below conditions on a length-3 vector `y`, but only references `y[1]` and `y[3]`.
The value supplied at `y[2]` is never used and has no effect on the result:

```julia
using RxInfer

@model function partial(y)
    x ~ NormalMeanVariance(0.0, 100.0)
    y[1] ~ NormalMeanVariance(x, 1.0)
    y[3] ~ NormalMeanVariance(x, 1.0)   # y[2] is never referenced
end

# y[2] = 999.0 is provided but ignored — only y[1] and y[3] inform the posterior.
result = infer(model = partial(), data = (y = [1.0, 999.0, 3.0],), iterations = 1)
```

The same applies to multi-dimensional tensors and to data tensors that are sliced and passed
into sub-models, which is the typical pattern for masked, per-time-step observation blocks:

```julia
@model function observe(y_, mask_, x)
    for j in eachindex(mask_)
        if mask_[j]
            y_[j] ~ NormalMeanVariance(x, 1.0)
        end
    end
end

@model function masked_chain(y, mask)
    J, T = size(mask)
    x ~ NormalMeanVariance(0.0, 100.0)
    for t in 1:T
        # `observe` only references the observed sensors of the t-th column of `y`.
        x ~ observe(y_ = y[1:J, t:t], mask_ = mask[1:J, t:t])
    end
end
```

## How the data is fed

When a conditioned data array is only partially referenced, the corresponding array of data
variables is **sparse** — it has the same bounding-box shape as the data, but only the
referenced indices are materialized. During inference RxInfer feeds the provided data into
those variables **by index**: the data variable at position `I` receives the value `data[I]`,
and entries of `data` at unreferenced positions are ignored. You therefore pass `data` with
its natural, dense shape — no need to pre-mask or reshape it to match the referenced subset.

This applies to both batch inference (`infer` with `data`) and streaming inference (`infer`
with `datastream`) — in the streaming case each tick's dense value is fed to the materialized
variables by the same index alignment.

## Non-standard (offset) data indexing

Conditioned data may use **non-standard indexing**, such as an `OffsetArray` whose axes start
at `0` (or at a negative index). RxInfer presents such data to the model with standard 1-based
axes — the values and their order are preserved, only the addressing is rebased. You therefore
write your model with ordinary 1-based indexing (`1:n`, `eachindex`, `axes`, …) regardless of
the data's native axes, and partial/sparse referencing works exactly as above:

```julia
using RxInfer, OffsetArrays

@model function partial(y)
    x ~ NormalMeanVariance(0.0, 100.0)
    y[1] ~ NormalMeanVariance(x, 1.0)
    y[3] ~ NormalMeanVariance(x, 1.0)
end

# A 0-based OffsetArray is accepted; the model still indexes 1-based.
ydata  = OffsetArray([1.0, 999.0, 3.0], 0:2)
result = infer(model = partial(), data = (y = ydata,), iterations = 1)
```

Standard (already 1-based) arrays are passed through unchanged, with no copy. Note that
indexing the model itself with a literal offset index (e.g. writing `y[0]` inside `@model`)
is **not** supported — variable arrays are 1-based; only the *data*'s indexing is rebased.

!!! note "Rebasing incurs a copy"
    Rebasing offset data to 1-based indexing allocates a copy of the array (once per `infer`
    call for batch inference; once per tick for the affected variable in streaming inference).
    A warning is emitted when this happens, gated by the `warn` keyword of [`infer`](@ref) (set
    `warn = false` to silence it). To avoid the copy entirely, pass a 1-based array — e.g.
    `collect(x)` or `OffsetArrays.no_offset_view(x)`. Standard 1-based data incurs no copy and no
    warning.

## Notes and caveats

- The provided `data` array must be indexable at every referenced position, i.e. its bounding
  box must contain all the indices your model touches. Extra (unreferenced) entries are fine.
- This applies to **data** inputs. Latent (random) variables are normally fully connected; a
  partially-referenced latent array would leave half-edges in the graph and is not a supported
  construction.
- Because unreferenced indices never become part of the graph, they incur no message-passing
  or memory cost — only the referenced sub-graph is built and run.
