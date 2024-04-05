# [Using methods from RxInfer](@id lib-using-methods)

In the Julia programming language (in contrast to Python for example) the most common way of loading a module is:

```julia
using RxInfer
```

A nice explanation about how modules/packages work in Julia can be found in [the official documentation](https://docs.julialang.org/en/v1/manual/modules/#Standalone-using-and-import).

In a nutshell, Julia automatically resolves all name collisions and there is no a lot of benefit of importing specific names, e.g.:

```julia
import RxInfer: mean
```

One of the reasons for that is that Julia uses multiple-dispatch capabilities to merge names automatically and will indicate (with a warning) if something went wrong or names have unresolvable collisions on types. As a small example of this feature consider the following small import example:

```@example import
import RxInfer: mean as mean_from_rxinfer
import Distributions: mean as mean_from_distributions

mean_from_rxinfer === mean_from_distributions
```

Even though we import `mean` function from two different packages they actually refer to the same object. Worth noting that this is not always the case - Julia will print a warning in case it finds unresolvable conflicts and usage of such functions will be disallowed unless user `import` them specifically. Read more about this in the [section of the Julia's documentation](https://docs.julialang.org/en/v1/manual/modules/#Handling-name-conflicts).

```@example another_import
# It is easier to let Julia resolve names automatically
# Julia will not overwrite `mean` that is coming from both packages
using RxInfer, Distributions 
```

```@example another_import
mean(Normal(0.0, 1.0)) # `Normal` is an object from `Distributions.jl`
```

```@example another_import
mean(NormalMeanVariance(0.0, 1.0)) # `NormalMeanVariance` is an object from `RxInfer.jl`
```

## [List of available methods](@id lib-list-methods)

Below you can find a list of **exported** methods from RxInfer.jl. All methods (even private) can be always accessed with `RxInfer.` prefix, e.g `RxInfer.mean`.

!!! note
    Some exported names are (for legacy reasons) intended for private usage only. As a result some of these methods do not have a proper associated documentation with them. We constantly improve RxInfer.jl library and continue to add better documentation for many exported methods, but a small portion of these methods could be removed from this list in the future.

```@example list
using RxInfer #hide
foreach(println, names(RxInfer))
```