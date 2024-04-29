export @autoupdates

import Base: isempty, show, map
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

macro autoupdates(expression)
    return esc(parse_autoupdates(:([]), expression))
end

macro autoupdates(options, expression)
    return esc(parse_autoupdates(options, expression))
end

function parse_autoupdates(options, expression)
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

    # If expression is not a function definition, then `funcdef` is `nothing`
    block, funcdef = if @capture(expression, function (fcall_ | fcall_)
        body_
    end)
        funcdef = MacroTools.splitdef(expression)
        funcdef[:body], funcdef
    else
        expression, nothing
    end

    if !(block isa Expr) || block.head !== :block
        error("Autoupdates requires a block of code `begin ... end` or a full function definition as an input")
    end

    autoupdate_check_reserved_expressions(block)
    specification = gensym(:autoupdate_specification)
    code = autoupdate_parse_autoupdate_specification_expr(specification, block)
    body = quote
        let $specification = RxInfer.AutoUpdateSpecification($warn, $strict, ())
            $code
            isempty($specification) && error("`@autoupdates` did not find any auto-updates specifications. Check the documentation for more information.")
            $specification
        end
    end

    if !isnothing(funcdef)
        funcdef[:body] = body
        MacroTools.combinedef(funcdef)
    else
        body
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
    return length(getspecifications(specification))
end

"Returns `true` if the auto-update specification is empty"
function Base.isempty(specification::AutoUpdateSpecification)
    return iszero(numautoupdates(specification))
end

"Returns the individual auto-update specification at the given index"
function getautoupdate(specification::AutoUpdateSpecification, index)
    return getindex(getspecifications(specification), index)
end

function addspecification(specification::AutoUpdateSpecification, labels, mapping)
    return addspecification(specification, getspecifications(specification), labels, mapping)
end

function addspecification(specification::AutoUpdateSpecification, specifications::Tuple, labels, mapping)
    return AutoUpdateSpecification(
        is_autoupdates_warn(specification), is_autoupdates_strict(specification), (specifications..., IndividualAutoUpdateSpecification(labels, mapping))
    )
end

function getspecifications(specification::AutoUpdateSpecification)
    return specification.specifications
end

function getvarlabels(specification::AutoUpdateSpecification)
    return mapreduce(getvarlabels, __reducevarlabels, getspecifications(specification); init = ())
end
# These functions are used to reduce the individual auto-update specifications to the list of variable labels
__reducevarlabels(collected, upcoming::Tuple) = (collected..., map(getlabel, upcoming)...)
__reducevarlabels(collected, upcoming) = (collected..., getlabel(upcoming))

function Base.map(f::F, specification::AutoUpdateSpecification) where {F}
    is_warn = is_autoupdates_warn(specification)
    is_strict = is_autoupdates_strict(specification)
    return AutoUpdateSpecification(is_warn, is_strict, map(f, getspecifications(specification)))
end

function Base.show(io::IO, specification::AutoUpdateSpecification)
    println(io, "@autoupdates begin")
    foreach(getspecifications(specification)) do spec
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
    AutoUpdateVariableLabel{L, I}(label, [ index = nothing ])

A structure that holds the label of the variable to update and its index.
By default, the index is set to `nothing`.
"""
struct AutoUpdateVariableLabel{L, I} end

getlabel(::AutoUpdateVariableLabel{L, I}) where {L, I} = L
getindex(::AutoUpdateVariableLabel{L, I}) where {L, I} = I

AutoUpdateVariableLabel(label::Symbol) = AutoUpdateVariableLabel{label, ()}()
AutoUpdateVariableLabel(label::Symbol, index::Tuple) = AutoUpdateVariableLabel{label, index}()

Base.show(io::IO, specification::AutoUpdateVariableLabel) =
    isempty(getindex(specification)) ? print(io, getlabel(specification)) : print(io, getlabel(specification), "[", join(getindex(specification), ", "), "]")

"""
    AutoUpdateMapping(arguments, mappingFn)

A structure that holds the arguments and the mapping function for the individual auto-update specification.
"""
struct AutoUpdateMapping{F, A}
    mappingFn::F
    arguments::A
end

getmappingfn(mapping::AutoUpdateMapping) = mapping.mappingFn
getarguments(mapping::AutoUpdateMapping) = mapping.arguments

Base.show(io::IO, mapping::AutoUpdateMapping) = print(io, getmappingfn(mapping), "(", join(getarguments(mapping), ", "), ")")

struct AutoUpdateFetchMarginalArgument{L, I} end

getlabel(::AutoUpdateFetchMarginalArgument{L, I}) where {L, I} = L
getindex(::AutoUpdateFetchMarginalArgument{L, I}) where {L, I} = I

AutoUpdateFetchMarginalArgument(label::Symbol) = AutoUpdateFetchMarginalArgument{label, ()}()
AutoUpdateFetchMarginalArgument(label::Symbol, index::Tuple) = AutoUpdateFetchMarginalArgument{label, index}()

Base.show(io::IO, argument::AutoUpdateFetchMarginalArgument) =
    isempty(getindex(argument)) ? print(io, "q(", getlabel(argument), ")") : print(io, "q(", getlabel(argument), "[", join(getindex(argument), ", "), "])")

struct AutoUpdateFetchMessageArgument{L, I} end

getlabel(::AutoUpdateFetchMessageArgument{L, I}) where {L, I} = L
getindex(::AutoUpdateFetchMessageArgument{L, I}) where {L, I} = I

AutoUpdateFetchMessageArgument(label::Symbol) = AutoUpdateFetchMessageArgument{label, ()}()
AutoUpdateFetchMessageArgument(label::Symbol, index::Tuple) = AutoUpdateFetchMessageArgument{label, index}()

Base.show(io::IO, argument::AutoUpdateFetchMessageArgument) =
    isempty(getindex(argument)) ? print(io, "μ(", getlabel(argument), ")") : print(io, "μ(", getlabel(argument), "[", join(getindex(argument), ", "), "])")

# Model interactions with GraphPPL generator

import GraphPPL

function check_model_generator_compatibility(specification::AutoUpdateSpecification, model::GraphPPL.ModelGenerator)
    kwargskeys = keys(GraphPPL.getkwargs(model))
    varlabels = getvarlabels(specification)
    for label in varlabels
        if label ∈ kwargskeys
            warnmsg = lazy"Autoupdates defines an update for `$label`, but `$label` has been reserved in the model as a constant. Use `warn = false` option to supress the warning. Use `strict = true` option to turn the warning into an error."
            if is_autoupdates_strict(specification)
                error(warnmsg)
            elseif is_autoupdates_warn(specification)
                @warn(warnmsg)
            end
        end
    end
    return nothing
end

function autoupdates_data_handlers(specification::AutoUpdateSpecification)
    varlabels = getvarlabels(specification)
    return NamedTuple{varlabels}(ntuple(_ -> DeferredDataHandler(), length(varlabels)))
end

"""
    prepare_autoupdates_for_model(autoupdates, model)

This function extracts the variables saved in the `autoupdates` from the model.
Replaces `AutoUpdateFetchMarginalArgument` and `AutoUpdateFetchMessageArgument` with actual streams.
"""
function prepare_autoupdates_for_model(autoupdates, model)
    vardict = getvardict(model)
    return map((autoupdate) -> prepare_individual_autoupdate_for_model(autoupdate, model, vardict), autoupdates)
end
function prepare_individual_autoupdate_for_model(autoupdate::IndividualAutoUpdateSpecification, model, vardict)
    return prepare_individual_autoupdate_for_model(getvarlabels(autoupdate), getmapping(autoupdate), model, vardict)
end
function prepare_individual_autoupdate_for_model(varlabels, mapping, model, vardict)
    prepared_varlabels = prepare_varlabels_autoupdate_for_model(varlabels, model, vardict)
    prepared_mapping = prepare_mapping_autoupdate_for_model(mapping, model, vardict)
    return IndividualAutoUpdateSpecification(prepared_varlabels, prepared_mapping)
end

function prepare_varlabels_autoupdate_for_model(varlabel::AutoUpdateVariableLabel, model, vardict)
    return prepare_varlabels_autoupdate_for_model(getlabel(varlabel), getindex(varlabel), model, vardict)
end
function prepare_varlabels_autoupdate_for_model(varlabels::Tuple, model, vardict)
    return map((l) -> prepare_varlabels_autoupdate_for_model(l, model, vardict), varlabels)
end
function prepare_varlabels_autoupdate_for_model(label::Symbol, index::Tuple{}, model, vardict) 
    haskey(vardict, label) || error(lazy"The `autoupdate` specification defines an update for `$(label)`, but the model has no variable named `$(label)`")
    return getvariable(vardict[label])
end
function prepare_varlabels_autoupdate_for_model(label::Symbol, index::Tuple, model, vardict) 
    haskey(vardict, label) || error(lazy"The `autoupdate` specification defines an update for `$(label)`, but the model has no variable named `$(label)`")
    return getvariable(vardict[label][index...])
end

function prepare_mapping_autoupdate_for_model(mapping::AutoUpdateMapping, model, vardict)
    prepared_arguments = map((a) -> prepare_mapping_argument_for_model(a, model, vardict), getarguments(mapping))
    return AutoUpdateMapping(getmappingfn(mapping), prepared_arguments)
end
prepare_mapping_argument_for_model(mapping::AutoUpdateMapping, model, vardict) = prepare_mapping_autoupdate_for_model(mapping, model, vardict)
prepare_mapping_argument_for_model(any::Any, model, vardict) = any

# Prepare expression of `q(_)`
function prepare_mapping_argument_for_model(marginal::AutoUpdateFetchMarginalArgument, model, vardict)
    label = getlabel(marginal)
    if !haskey(vardict, label)
        error(lazy"The `autoupdate` specification defines an update from `q($(label))`, but the model has no variable named `$(label)`")
    end
    return prepare_mapping_argument_for_model(marginal, label, getindex(marginal), model, vardict)
end
prepare_mapping_argument_for_model(::AutoUpdateFetchMarginalArgument, label::Symbol, index::Tuple{}, model, vardict) = getmarginal(vardict[label], IncludeAll())
prepare_mapping_argument_for_model(::AutoUpdateFetchMarginalArgument, label::Symbol, index::Tuple, model, vardict) = getmarginal(vardict[label][index...], IncludeAll())

# Prepare expression of `μ(_)`
function prepare_mapping_argument_for_model(message::AutoUpdateFetchMessageArgument, model, vardict)
    label = getlabel(message)
    if !haskey(vardict, label)
        error(lazy"The `autoupdate` specification defines an update from `μ($(label))`, but the model has no variable named `$(label)`")
    end
    return prepare_mapping_argument_for_model(message, label, getindex(message), model, vardict)
end
function prepare_mapping_argument_for_model(::AutoUpdateFetchMessageArgument, label::Symbol, index::Tuple{}, model, vardict)
    variable = vardict[label]
    return messageout(variable, degree(variable))
end
function prepare_mapping_argument_for_model(::AutoUpdateFetchMessageArgument, label::Symbol, index::Tuple, model, vardict)
    variable = vardict[label][index...]
    return messageout(variable, degree(variable))
end

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
