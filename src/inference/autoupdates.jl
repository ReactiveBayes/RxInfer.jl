export @autoupdates

import Base: isempty, show, map
import MacroTools
import MacroTools: @capture

"""
    @autoupdates [ options... ] begin 
        argument_to_update = some_function(q(some_variable_from_the_model))
    end

Creates the auto-updates specification for the `infer` function for the online-streaming Bayesian inference procedure, where 
it is important to update prior states based on the new updated posteriors. Read more information about the 
`@autoupdates` syntax in the official documentation.
"""
macro autoupdates end

macro autoupdates(expression)
    return esc(parse_autoupdates(:([]), expression))
end

macro autoupdates(options, expression)
    return esc(parse_autoupdates(options, expression))
end

"""
    parse_autoupdates(options, expression)

Parses the internals of the expression passed to the `@autoupdates` macro and returns the `RxInfer.AutoUpdateSpecification` structure.
"""
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

"""
    autoupdate_check_reserved_expressions(block)

This function checks if the expression is a valid autoupdate specification
some expressions are forbidden within the autoupdate specification.
"""
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

const EmptyAutoUpdateSpecification = AutoUpdateSpecification(true, true, ())

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

"Appends the individual auto-update specification to the existing specification"
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

"Returns the labels of the auto-update specification, which are the names of the variables to update"
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

"This autoupdate would fetch updates from the marginal of a variable"
struct AutoUpdateFetchMarginalArgument{L, I} end

getlabel(::AutoUpdateFetchMarginalArgument{L, I}) where {L, I} = L
getindex(::AutoUpdateFetchMarginalArgument{L, I}) where {L, I} = I

AutoUpdateFetchMarginalArgument(label::Symbol) = AutoUpdateFetchMarginalArgument{label, ()}()
AutoUpdateFetchMarginalArgument(label::Symbol, index::Tuple) = AutoUpdateFetchMarginalArgument{label, index}()

Base.show(io::IO, argument::AutoUpdateFetchMarginalArgument) =
    isempty(getindex(argument)) ? print(io, "q(", getlabel(argument), ")") : print(io, "q(", getlabel(argument), "[", join(getindex(argument), ", "), "])")

"This autoupdate would fetch updates from the last message (in the array of messages) of a variable"
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
    # return NamedTuple{varlabels}(ntuple(_ -> DeferredDataHandler(), length(varlabels)))
    # Below is similar to the commented version, but supports duplicate entries, e.g if a user
    # writes `ins[1], ins[2] = ...` in the `@autoupdates` block
    return reduce(varlabels; init = (;)) do ntuple, label
        append = NamedTuple{(label,)}((DeferredDataHandler(),))
        return (; ntuple..., append...)
    end
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

import Rocket: getrecent

struct FetchRecentArgument{L, S}
    stream::S
end

FetchRecentArgument(label::Symbol, stream::S) where {S} = FetchRecentArgument{label, S}(stream)

getlabel(::FetchRecentArgument{L}) where {L} = L
Rocket.getrecent(argument::FetchRecentArgument) = getrecent(argument, argument.stream)
Rocket.getrecent(argument::FetchRecentArgument, stream) = getrecent(stream)
Rocket.getrecent(argument::FetchRecentArgument, streams::AbstractArray) = map(stream -> getrecent(argument, stream), streams)

# Prepare expression of `q(_)`
function prepare_mapping_argument_for_model(marginal::AutoUpdateFetchMarginalArgument, model, vardict)
    label = getlabel(marginal)
    if !haskey(vardict, label)
        error(lazy"The `autoupdate` specification defines an update from `q($(label))`, but the model has no variable named `$(label)`")
    end
    index = getindex(marginal)
    var   = isempty(index) ? vardict[label] : vardict[label][index...]
    return FetchRecentArgument(label, _marginal_argument(var))
end
_marginal_argument(variable) = getmarginal(variable, IncludeAll())
_marginal_argument(variables::AbstractArray) = map(_marginal_argument, variables)

# Prepare expression of `μ(_)`
function prepare_mapping_argument_for_model(message::AutoUpdateFetchMessageArgument, model, vardict)
    label = getlabel(message)
    if !haskey(vardict, label)
        error(lazy"The `autoupdate` specification defines an update from `μ($(label))`, but the model has no variable named `$(label)`")
    end
    index = getindex(marginal)
    var   = isempty(index) ? vardict[label] : vardict[label][index...]
    return FetchRecentArgument(label, _message_argument(var))
end
_message_argument(variable) = messageout(variable, degree(variable))
_message_argument(variables::AbstractArray) = map(_message_argument, variables)

import Base: fetch

function Base.fetch(autoupdate::IndividualAutoUpdateSpecification)
    return autoupdate_mapping_fetch(getmapping(autoupdate))
end

function Base.fetch(mapping::AutoUpdateMapping)
    return autoupdate_mapping_fetch(mapping)
end
autoupdate_mapping_fetch(mapping::AutoUpdateMapping) = getmappingfn(mapping)(map(autoupdate_mapping_fetch, getarguments(mapping))...)
autoupdate_mapping_fetch(any) = any
autoupdate_mapping_fetch(argument::FetchRecentArgument) = autoupdate_mapping_fetch(argument, Rocket.getrecent(argument))
autoupdate_mapping_fetch(argument::FetchRecentArgument, something) = something
autoupdate_mapping_fetch(argument::FetchRecentArgument, ::Nothing) =
    error("The initial value for `$(getlabel(argument))` has not been specified, but is required in the `@autoupdates`.")

function run_autoupdate!(autoupdates::AutoUpdateSpecification)
    return run_autoupdate!(getspecifications(autoupdates), map(fetch, getspecifications(autoupdates)))
end

function run_autoupdate!(specifications::Tuple, prefetched::Tuple)
    length(specifications) === length(prefetched) || error("Cannot execute autoupdate. The number of specifications and prefetched values must be equal.")
    foreach(zip(specifications, prefetched)) do (specification, update)
        varlabels = getvarlabels(specification)
        varlabels_tupled = as_tuple(varlabels)
        update_tupled = as_tuple(update)

        if !isequal(length(varlabels_tupled), length(update_tupled))
            error("Cannot run autoupdate for `$(varlabels_tupled)`. The update provides `$(length(update_tuple))` values, but `$(length(varlabels_tupled))` is needed.")
        end

        foreach(zip(varlabels_tupled, update_tupled)) do (var, val)
            update!(var, val)
        end
    end
end
