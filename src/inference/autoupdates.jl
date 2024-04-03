export @autoupdates

struct FromMarginalAutoUpdate end
struct FromMessageAutoUpdate end

import Base: string

Base.string(::FromMarginalAutoUpdate) = "q"
Base.string(::FromMessageAutoUpdate) = "μ"

import Base: fetch

Base.fetch(strategy::Union{FromMarginalAutoUpdate, FromMessageAutoUpdate}, variables::AbstractArray) = (Base.fetch(strategy, variable) for variable in variables)
Base.fetch(::FromMarginalAutoUpdate, variable::Union{DataVariable, RandomVariable}) = ReactiveMP.getmarginal(variable, IncludeAll())
Base.fetch(::FromMessageAutoUpdate, variable::RandomVariable) = ReactiveMP.messagein(variable, 1) # Here we assume that predictive message has index `1`
Base.fetch(::FromMessageAutoUpdate, variable::DataVariable) = error("`FromMessageAutoUpdate` fetch strategy is not implemented for `DataVariable`")

struct RxInferenceAutoUpdateIndexedVariable{V, I}
    variable :: V
    index    :: I
end

Base.string(indexed::RxInferenceAutoUpdateIndexedVariable) = string(indexed.variable, "[", join(indexed.index, ", "), "]")

hasdatavar(model, variable::RxInferenceAutoUpdateIndexedVariable)   = hasdatavar(model, variable.variable)
hasrandomvar(model, variable::RxInferenceAutoUpdateIndexedVariable) = hasrandomvar(model, variable.variable)

function Base.getindex(model::ProbabilisticModel, indexed::RxInferenceAutoUpdateIndexedVariable)
    return model[indexed.variable][indexed.index...]
end

struct RxInferenceAutoUpdateSpecification{N, F, C, V}
    labels   :: NTuple{N, Symbol}
    from     :: F
    callback :: C
    variable :: V
end

getlabels(specification::RxInferenceAutoUpdateSpecification) = specification.labels

function Base.show(io::IO, specification::RxInferenceAutoUpdateSpecification)
    print(io, join(specification.labels, ","), " = ", string(specification.callback), "(", string(specification.from), "(", string(specification.variable), "))")
end

function (specification::RxInferenceAutoUpdateSpecification)(vardict)
    datavars = map(specification.labels) do label
        haskey(vardict, label) || error("Autoupdate specification defines an update for `$(label)`, but the model has no variable named `$(label)`")
        return getvariable(vardict[label])
    end

    (haskey(vardict, specification.variable)) ||
        error("Autoupdate specification defines an update from `$(specification.variable)`, but the model has no variable named `$(specification.variable)`")

    variable = getvariable(vardict[specification.variable])

    return RxInferenceAutoUpdate(specification.variable, datavars, specification.callback, fetch(specification.from, variable))
end

struct RxInferenceAutoUpdate{L, N, C, R}
    varlabel :: L
    datavars :: N
    callback :: C
    recent   :: R
end

getvarlabel(autoupdate::RxInferenceAutoUpdate) = autoupdate.varlabel

import Base: fetch

Base.fetch(autoupdate::RxInferenceAutoUpdate) = fetch(autoupdate, autoupdate.recent)
Base.fetch(autoupdate::RxInferenceAutoUpdate, something) = fetch(autoupdate, something, ReactiveMP.getdata(ReactiveMP.getrecent(something)))
Base.fetch(autoupdate::RxInferenceAutoUpdate, something::Union{AbstractArray, Base.Generator}) = fetch(autoupdate, something, ReactiveMP.getdata.(ReactiveMP.getrecent.(something)))

Base.fetch(autoupdate::RxInferenceAutoUpdate, _, data) = zip(as_tuple(autoupdate.datavars), as_tuple(autoupdate.callback(data)))
Base.fetch(autoupdate::RxInferenceAutoUpdate, _, data::Nothing) =
    error("The initial value for `$(autoupdate.varlabel)` in the `@autoupdates` has not been specified. Consider using `initmarginals` or `initmessages`.")

import MacroTools
import MacroTools: @capture

"""
    @autoupdates

Creates the auto-updates specification for the `rxinference` function. In the online-streaming Bayesian inference procedure it is important to update your priors for the future 
states based on the new updated posteriors. The `@autoupdates` structure simplify such a specification. It accepts a single block of code where each line defines how to update 
the `datavar`'s in the probabilistic model specification. 

Each line of code in the auto-update specification defines `datavar`s, which need to be updated, on the left hand side of the equality expression and the update function on the right hand side of the expression.
The update function operates on posterior marginals in the form of the `q(symbol)` expression.

For example:

```julia
@autoupdates begin 
    x = f(q(z))
end
```

This structure specifies to automatically update `x = datavar(...)` as soon as the inference engine computes new posterior over `z` variable. It then applies the `f` function
to the new posterior and calls `update!(x, ...)` automatically. 

As an example consider the following model and auto-update specification:

```julia
@model function kalman_filter()
    x_current_mean = datavar(Float64)
    x_current_var  = datavar(Float64)

    x_current ~ Normal(mean = x_current_mean, var = x_current_var)

    x_next ~ Normal(mean = x_current, var = 1.0)

    y = datavar(Float64)
    y ~ Normal(mean = x_next, var = 1.0)
end
```

This model has two `datavar`s that represent our prior knowledge of the `x_current` state of the system. The `x_next` random variable represent the next state of the system that 
is connected to the observed variable `y`. The auto-update specification could look like:

```julia
autoupdates = @autoupdates begin
    x_current_mean, x_current_var = mean_cov(q(x_next))
end
```

This structure specifies to update our prior as soon as we have a new posterior `q(x_next)`. It then applies the `mean_cov` function on the updated posteriors and updates 
`datavar`s `x_current_mean` and `x_current_var` automatically.

See also: [`infer`](@ref)
"""
macro autoupdates(code)
    ((code isa Expr) && (code.head === :block)) || error("Autoupdate requires a block of code `begin ... end` as an input")

    specifications = []

    code = MacroTools.postwalk(code) do expression
        # We modify all expression of the form `... = callback(q(...))` or `... = callback(μ(...))`
        if @capture(expression, (lhs_ = callback_(rhs_)) | (lhs_ = callback_(rhs__)))
            if @capture(rhs, (q(variable_)) | (μ(variable_)))
                # First we check that `variable` is a plain Symbol or an index operation
                if (variable isa Symbol)
                    variable = QuoteNode(variable)
                elseif (variable isa Expr) && (variable.head === :ref)
                    variable = :(RxInfer.RxInferenceAutoUpdateIndexedVariable($(QuoteNode(variable.args[1])), ($(variable.args[2:end])...,)))
                else
                    error("Variable in the expression `$(expression)` must be a plain name or and indexing operation, but a complex expression `$(variable)` found.")
                end
                # Next we extract `datavars` specification from the `lhs`                    
                datavars = if lhs isa Symbol
                    (lhs,)
                elseif lhs isa Expr && lhs.head === :tuple && all(arg -> arg isa Symbol, lhs.args)
                    Tuple(lhs.args)
                else
                    error("Left hand side of the expression `$(expression)` must be a single symbol or a tuple of symbols")
                end
                # Only two options are possible within this `if` block
                from = @capture(rhs, q(smth_)) ? :(RxInfer.FromMarginalAutoUpdate()) : :(RxInfer.FromMessageAutoUpdate())

                push!(specifications, :(RxInfer.RxInferenceAutoUpdateSpecification($(datavars...,), $from, $callback, $variable)))

                return :(nothing)
            else
                error("Complex call expression `$(expression)` in the `@autoupdates` macro")
            end
        else
            return expression
        end
    end

    isempty(specifications) && error("`@autoupdates` did not find any auto-updates specifications. Check the documentation for more information.")

    output = quote
        begin
            $code

            ($(specifications...),)
        end
    end

    return esc(output)
end
