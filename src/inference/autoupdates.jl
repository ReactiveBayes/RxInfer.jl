export @autoupdates

import Base: isempty, show
import MacroTools
import MacroTools: @capture

"""
    @autoupdates [ options... ] begin 
        ...
    end

Creates the auto-updates specification for the `infer` function. In the online-streaming Bayesian inference procedure it is important to update your priors for the 
states based on the new updated posteriors. The `@autoupdates` structure simplify such a specification. The `@autoupdates` macro detects lines of code of the following structure
```
(labels...) = f(arguments...)
```
Checks if `arguments...` has either `q(_)` in its sub-expressions and adds such expressions to the specification list. 
All other expressions are left untouched. The result of the macro execution is the `RxInfer.AutoUpdateSpecification` structure that holds the collection 
of individual auto-update specifications.

Each individualauto-update specification refers to model's arguments, which need to be updated, on the left hand side of the equality expression and 
the update function on the right hand side of the expression.
The update function operates on posterior marginals in the form of the `q(symbol)` expression.

For example:

```julia
@autoupdates begin 
    x = mean(q(z))
end
```

This structure specifies to automatically update argument `x` as soon as the inference engine computes new posterior over `z` variable.
It then applies the `mean` function to the new posterior and updates the value of `x` automatically. 

As another example consider the following model and auto-update specification:

```julia
@model function kalman_filter(y, x_current_mean, x_current_var)
    x_current ~ Normal(mean = x_current_mean, var = x_current_var)
    x_next    ~ Normal(mean = x_current, var = 1.0)
    y         ~ Normal(mean = x_next, var = 1.0)
end
```

This model has two arguments that represent our prior knowledge of the `x_current` state of the system. 
The `x_next` random variable represent the next state of the system that 
is connected to the observed variable `y`. The auto-update specification could look like:

```julia
autoupdates = @autoupdates begin
    x_current_mean, x_current_var = mean_var(q(x_next))
end
```
This structure specifies to update our prior as soon as we have a new posterior `q(x_next)`. It then applies the `mean_var` function on the 
updated posteriors and updates `x_current_mean` and `x_current_var` automatically.

More complex `@autoupdates` are also allowed. For example, the following code is a valid `@autoupdates` specification:
```julia
@autoupdates begin 
    x = clamp(mean(q(z)), 0, 1)
end
```

The `@autoupdates` macro accepts an optional set of `[ options... ]` before the `begin ... end` block. The available options are:
- `warn = true/false`: Enables or disables warnings when with incomaptible model. Set to `true` by default.
- `strict = true/false`: Turns warnings into errors. Set to `false` by default.

See also: [`infer`](@ref)
"""
macro autoupdates end

macro autoupdates(block)
    return esc(parse_autoupdates(:([]), block))
end

macro autoupdates(options, block)
    return esc(parse_autoupdates(options, block))
end

function parse_autoupdates(options, block)
    if !@capture(options, [optionargs__])
        error("Invalid options for `@autoupdates`. Expected `[ options... ]`")
    end

    warn::Bool = true
    strict::Bool = false

    for optionarg in optionargs
        if @capture(optionarg, warn = val_)
            val isa Bool || error("Invalid value for `warn` option. Expected `true` or `false`.")
            warn = val::Bool
        elseif @capture(optionarg, strict = val_)
            val isa Bool || error("Invalid value for `strict` option. Expected `true` or `false`.")
            strict = val::Bool
        else
            error("Unknown option for `@autoupdates`: $optionarg. Supported options are [ `warn`, `strict` ].")
        end
    end

    if !(block isa Expr) || block.head !== :block
        error("Autoupdates requires a block of code `begin ... end` as an input")
    end
    autoupdate_check_reserved_expressions(block)
    specification = gensym(:autoupdate_specification)
    code = autoupdate_parse_autoupdate_specification_expr(specification, block)
    return quote
        let $specification = RxInfer.AutoUpdateSpecification($warn, $strict, ())
            $code
            isempty($specification) && error("`@autoupdates` did not find any auto-updates specifications. Check the documentation for more information.")
            $specification
        end
    end
end

# This function checks if the expression is a valid autoupdate specification
# some expressions are forbidden within the autoupdate specification
function autoupdate_check_reserved_expressions(block)
    return MacroTools.postwalk(block) do subexpr
        if @capture(subexpr, (q(_) = _) | (μ(_) = _))
            error("Cannot evaluate the following subexpression $(subexpr). q(x) and μ(x) are reserved functions")
        end
        return subexpr
    end
end

# This function checks if the expression is a valid autoupdate specification
# it does that by checking if the expression has the following sub-expression:
# - q(_)
# - μ(_)
function autoupdate_argument_inexpr(expr)
    result = false
    MacroTools.postwalk(expr) do subexpr
        if @capture(subexpr, (q(arg_)) | (μ(arg_)))
            result = true
        end
        return subexpr
    end
    return result
end

# This function checks if the expression is a valid autoupdate mapping
function is_autoupdate_mapping_expr(expr)
    return @capture(expr, f_(args__)) && any(autoupdate_argument_inexpr, args)
end

# This function converts an expression of the form `f(args...)` to the `AutoUpdateMapping` structure
function autoupdate_convert_mapping_expr(expr)
    return MacroTools.prewalk(expr) do subexpr
        if is_autoupdate_mapping_expr(subexpr)
            @capture(subexpr, f_(args__)) || error("Invalid autoupdate mapping expression: $subexpr")
            return :(RxInfer.AutoUpdateMapping($f, ($(map(autoupdate_convert_mapping_arg_expr, args)...),)))
        end
        return subexpr
    end
end

function autoupdate_convert_mapping_arg_expr(expr)
    if @capture(expr, q(s_[i__]))
        return :(RxInfer.AutoUpdateFetchMarginalArgument($(QuoteNode(s)), ($(i...),)))
    elseif @capture(expr, μ(s_[i__]))
        return :(RxInfer.AutoUpdateFetchMessageArgument($(QuoteNode(s)), ($(i...),)))
    elseif @capture(expr, q(s_))
        return :(RxInfer.AutoUpdateFetchMarginalArgument($(QuoteNode(s))))
    elseif @capture(expr, μ(s_))
        return :(RxInfer.AutoUpdateFetchMessageArgument($(QuoteNode(s))))
    else
        return expr
    end
end

function autoupdate_convert_labels_expr(expr)
    if expr isa Symbol
        return :(RxInfer.AutoUpdateVariableLabel($(QuoteNode(expr))))
    elseif @capture(expr, s_[indices__])
        return :(RxInfer.AutoUpdateVariableLabel($(QuoteNode(s)), ($(indices...),)))
    elseif @capture(expr, (s__,))
        return :(($(map(autoupdate_convert_labels_expr, s)...),))
    end
    @capture(expr, (s_Symbol) | ((s__,))) || error("Cannot create variable label from expression `$expr`")
    return nothing
end

function autoupdate_parse_autoupdate_specification_expr(spec, expr)
    return MacroTools.postwalk(expr) do subexpr
        if @capture(subexpr, lhs_ = rhs_) && is_autoupdate_mapping_expr(rhs)
            mapping = autoupdate_convert_mapping_expr(rhs)
            labels = autoupdate_convert_labels_expr(lhs)
            return :($spec = RxInfer.addspecification($spec, $labels, $mapping))
        end
        return subexpr
    end
end

"""
    AutoUpdateSpecification(specifications)

A structure that holds a collection of individual auto-update specifications. 
Each specification defines how to update the model's arguments 
based on the new posterior/messages updates. 
"""
struct AutoUpdateSpecification{S}
    warn::Bool
    strict::Bool
    specifications::S
end

is_autoupdates_warn(specification::AutoUpdateSpecification) = specification.warn
is_autoupdates_strict(specification::AutoUpdateSpecification) = specification.strict

"Returns the number of auto-updates in the specification"
function numautoupdates(specification::AutoUpdateSpecification)
    return length(specification.specifications)
end

"Returns `true` if the auto-update specification is empty"
function Base.isempty(specification::AutoUpdateSpecification)
    return iszero(numautoupdates(specification))
end

"Returns the individual auto-update specification at the given index"
function getautoupdate(specification::AutoUpdateSpecification, index)
    return specification.specifications[index]
end

function addspecification(specification::AutoUpdateSpecification, labels, mapping)
    return addspecification(specification, specification.specifications, labels, mapping)
end

function addspecification(specification::AutoUpdateSpecification, specifications::Tuple, labels, mapping)
    return AutoUpdateSpecification(
        is_autoupdates_warn(specification), is_autoupdates_strict(specification), (specifications..., IndividualAutoUpdateSpecification(labels, mapping))
    )
end

function getvarlabels(specification::AutoUpdateSpecification)
    return mapreduce(getvarlabels, __reducevarlabels, specification.specifications; init = ())
end
# These functions are used to reduce the individual auto-update specifications to the list of variable labels
__reducevarlabels(collected, upcoming::NTuple{N}) where {N} = (collected..., map(getlabel, upcoming)...)
__reducevarlabels(collected, upcoming) = (collected..., getlabel(upcoming))

function Base.show(io::IO, specification::AutoUpdateSpecification)
    println(io, "@autoupdates begin")
    foreach(specification.specifications) do spec
        println(io, "    ", spec)
    end
    println(io, "end")
end

"""
    IndividualAutoUpdateSpecification(varlabels, arguments, mapping)

A structure that defines how to update a single variable in the model.
It consists of the variable labels and the mapping function.
"""
struct IndividualAutoUpdateSpecification{L, M}
    varlabels::L
    mapping::M
end

"Returns the labels of the auto-update specification, which are the names of the variables to update"
getvarlabels(specification::IndividualAutoUpdateSpecification) = specification.varlabels

"Returns the mapping function of the auto-update specification, which defines how to update the variable"
getmapping(specification::IndividualAutoUpdateSpecification) = specification.mapping

Base.show(io::IO, specification::IndividualAutoUpdateSpecification) = print(io, getvarlabels(specification), " = ", getmapping(specification))

"""
    AutoUpdateVariableLabel{I}(label, [ index = nothing ])

A structure that holds the label of the variable to update and its index.
By default, the index is set to `nothing`.
"""
struct AutoUpdateVariableLabel{I}
    label::Symbol
    index::I
end

getlabel(label::AutoUpdateVariableLabel) = label.label
getindex(label::AutoUpdateVariableLabel) = label.index

AutoUpdateVariableLabel(label::Symbol) = AutoUpdateVariableLabel(label, nothing)

Base.show(io::IO, specification::AutoUpdateVariableLabel) =
    isnothing(specification.index) ? print(io, specification.label) : print(io, specification.label, "[", join(specification.index, ", "), "]")

"""
    AutoUpdateMapping(arguments, mappingFn)

A structure that holds the arguments and the mapping function for the individual auto-update specification.
"""
struct AutoUpdateMapping{F, A}
    mappingFn::F
    arguments::A
end

Base.show(io::IO, mapping::AutoUpdateMapping) = print(io, mapping.mappingFn, "(", join(mapping.arguments, ", "), ")")

struct AutoUpdateFetchMarginalArgument{I}
    label::Symbol
    index::I
end

AutoUpdateFetchMarginalArgument(label::Symbol) = AutoUpdateFetchMarginalArgument(label, nothing)

Base.show(io::IO, argument::AutoUpdateFetchMarginalArgument) =
    isnothing(argument.index) ? print(io, "q(", argument.label, ")") : print(io, "q(", argument.label, "[", join(argument.index, ", "), "])")

struct AutoUpdateFetchMessageArgument{I}
    label::Symbol
    index::I
end

AutoUpdateFetchMessageArgument(label::Symbol) = AutoUpdateFetchMessageArgument(label, nothing)

Base.show(io::IO, argument::AutoUpdateFetchMessageArgument) =
    isnothing(argument.index) ? print(io, "μ(", argument.label, ")") : print(io, "μ(", argument.label, "[", join(argument.index, ", "), "])")

# import Base: fetch

# Base.fetch(strategy::Union{FromMarginalAutoUpdate, FromMessageAutoUpdate}, variables::AbstractArray) = (Base.fetch(strategy, variable) for variable in variables)
# Base.fetch(::FromMarginalAutoUpdate, variable::Union{DataVariable, RandomVariable}) = ReactiveMP.getmarginal(variable, IncludeAll())
# Base.fetch(::FromMessageAutoUpdate, variable::RandomVariable) = ReactiveMP.messagein(variable, 1) # Here we assume that predictive message has index `1`
# Base.fetch(::FromMessageAutoUpdate, variable::DataVariable) = error("`FromMessageAutoUpdate` fetch strategy is not implemented for `DataVariable`")

# struct RxInferenceAutoUpdateIndexedVariable{V, I}
#     variable :: V
#     index    :: I
# end

# Base.string(indexed::RxInferenceAutoUpdateIndexedVariable) = string(indexed.variable, "[", join(indexed.index, ", "), "]")

# hasdatavar(model, variable::RxInferenceAutoUpdateIndexedVariable)   = hasdatavar(model, variable.variable)
# hasrandomvar(model, variable::RxInferenceAutoUpdateIndexedVariable) = hasrandomvar(model, variable.variable)

# function Base.getindex(model::ProbabilisticModel, indexed::RxInferenceAutoUpdateIndexedVariable)
#     return model[indexed.variable][indexed.index...]
# end

# struct RxInferenceAutoUpdateSpecification{N, F, C, V}
#     labels   :: NTuple{N, Symbol}
#     from     :: F
#     callback :: C
#     variable :: V
# end

# getlabels(specification::RxInferenceAutoUpdateSpecification) = specification.labels

# function Base.show(io::IO, specification::RxInferenceAutoUpdateSpecification)
#     print(io, join(specification.labels, ","), " = ", string(specification.callback), "(", string(specification.from), "(", string(specification.variable), "))")
# end

# function (specification::RxInferenceAutoUpdateSpecification)(vardict)
#     datavars = map(specification.labels) do label
#         haskey(vardict, label) || error("Autoupdate specification defines an update for `$(label)`, but the model has no variable named `$(label)`")
#         return getvariable(vardict[label])
#     end

#     (haskey(vardict, specification.variable)) ||
#         error("Autoupdate specification defines an update from `$(specification.variable)`, but the model has no variable named `$(specification.variable)`")

#     variable = getvariable(vardict[specification.variable])

#     return RxInferenceAutoUpdate(specification.variable, datavars, specification.callback, fetch(specification.from, variable))
# end

# struct RxInferenceAutoUpdate{L, N, C, R}
#     varlabel :: L
#     datavars :: N
#     callback :: C
#     recent   :: R
# end

# getvarlabel(autoupdate::RxInferenceAutoUpdate) = autoupdate.varlabel

# import Base: fetch

# Base.fetch(autoupdate::RxInferenceAutoUpdate) = fetch(autoupdate, autoupdate.recent)
# Base.fetch(autoupdate::RxInferenceAutoUpdate, something) = fetch(autoupdate, something, ReactiveMP.getdata(ReactiveMP.getrecent(something)))
# Base.fetch(autoupdate::RxInferenceAutoUpdate, something::Union{AbstractArray, Base.Generator}) = fetch(autoupdate, something, ReactiveMP.getdata.(ReactiveMP.getrecent.(something)))

# Base.fetch(autoupdate::RxInferenceAutoUpdate, _, data) = zip(as_tuple(autoupdate.datavars), as_tuple(autoupdate.callback(data)))
# Base.fetch(autoupdate::RxInferenceAutoUpdate, _, data::Nothing) =
#     error("The initial value for `$(autoupdate.varlabel)` has not been specified, but is required in the `@autoupdates`.")
