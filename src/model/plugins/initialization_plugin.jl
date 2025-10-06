export @initialization

# TODO (@wouterwln) This file is essentially a copy of the init plugin from `GraphPPL`, so we should consider a general solution for this that can be used in both cases.

using GraphPPL
import GraphPPL: IndexedVariable, unroll, children, fform, Model, Context, NodeLabel
using MacroTools

struct InitMessage end
struct InitMarginal end

struct InitDescriptor{S <: Union{InitMessage, InitMarginal}}
    type::S
    var_descriptor::IndexedVariable
end

getvardescriptor(m::InitDescriptor) = m.var_descriptor

Base.show(io::IO, m::InitDescriptor) = print(io, m.var_descriptor)

struct InitObject{S <: InitDescriptor, T}
    var_descriptor::S
    init_info::T
end

getvardescriptor(m::InitObject) = m.var_descriptor
getinitinfo(m::InitObject) = m.init_info

function Base.show(io::IO, m::InitObject{S, T}) where {S <: InitDescriptor{InitMarginal}, T}
    print(io, "q($(getvardescriptor(m))) = ")
    print(io, getinitinfo(m))
end

function Base.show(io::IO, m::InitObject{S, T}) where {S <: InitDescriptor{InitMessage}, T}
    print(io, "μ($(getvardescriptor(m))) = ")
    print(io, getinitinfo(m))
end

struct InitSpecification
    init_objects::Vector
    submodel_init::Vector
end

InitSpecification() = InitSpecification([], [])

function Base.show(io::IO, c::InitSpecification)
    indent = get(io, :indent, 1)
    head = get(io, :head, true)
    if head
        print(io, "Initial state: \n")
    else
        print(io, "\n")
    end
    for init in getinitobjects(c)
        print(io, "  "^indent)
        print(io, init)
        print(io, "\n")
    end
    for submodel in getsubmodelinit(c)
        print(io, "  "^indent)
        print(io, submodel)
        print(io, "\n")
    end
end

getinitobjects(m::InitSpecification) = m.init_objects
getsubmodelinit(m::InitSpecification) = m.submodel_init
getspecificsubmodelinit(m::InitSpecification) = filter(m -> is_specificsubmodelinit(m), getsubmodelinit(m))
getgeneralsubmodelinit(m::InitSpecification) = filter(m -> is_generalsubmodelinit(m), getsubmodelinit(m))

# TODO experiment with `findfirst` instead of `get` in benchmarks
getspecificsubmodelinit(m::InitSpecification, tag::Any) = get(filter(m -> getsubmodel(m) == tag, getsubmodelinit(m)), 1, nothing)
getgeneralsubmodelinit(m::InitSpecification, fform::Any) = get(filter(m -> getsubmodel(m) == fform, getsubmodelinit(m)), 1, nothing)

struct SpecificSubModelInit
    tag::GraphPPL.FactorID
    init_objects::InitSpecification
end

getsubmodel(c::SpecificSubModelInit) = c.tag
getinitobjects(c::SpecificSubModelInit) = c.init_objects
Base.push!(m::SpecificSubModelInit, o) = push!(m.init_objects, o)
SpecificSubModelInit(tag::GraphPPL.FactorID) = SpecificSubModelInit(tag, InitSpecification())
is_specificsubmodelinit(m::SpecificSubModelInit) = true
is_specificsubmodelinit(m) = false
getkey(m::SpecificSubModelInit) = getsubmodel(m)

struct GeneralSubModelInit
    fform::Any
    init_objects::InitSpecification
end

getsubmodel(c::GeneralSubModelInit) = c.fform
getinitobjects(c::GeneralSubModelInit) = c.init_objects
Base.push!(m::GeneralSubModelInit, o) = push!(m.init_objects, o)
GeneralSubModelInit(fform::Any) = GeneralSubModelInit(fform, InitSpecification())
is_generalsubmodelinit(m::GeneralSubModelInit) = true
is_generalsubmodelinit(m) = false
getkey(m::GeneralSubModelInit) = getsubmodel(m)

const SubModelInit = Union{GeneralSubModelInit, SpecificSubModelInit}

function Base.show(io::IO, init::SubModelInit)
    print(IOContext(io, (:indent => get(io, :indent, 0) + 2), (:head => false)), "Init for submodel ", getsubmodel(init), " = ", getinitobjects(init))
end

function Base.push!(m::InitSpecification, o::InitObject)
    if getvardescriptor(o) ∈ getvardescriptor.(getinitobjects(m))
        @warn "Variable $(getvardescriptor(getvardescriptor(o))) is initialized multiple times. The last initialization will be used."
        filter!(x -> getvardescriptor(getvardescriptor(x)) ≠ getvardescriptor(getvardescriptor(o)), m.init_objects)
    end
    push!(m.init_objects, o)
end
Base.push!(m::InitSpecification, o::SubModelInit) = push!(m.submodel_init, o)

default_init(any) = EmptyInit

function apply_init!(model::Model, init::InitSpecification)
    apply_init!(model, GraphPPL.get_principal_submodel(model), init)
end

function apply_init!(model::Model, context::Context, init::InitSpecification)
    for init_obj in getinitobjects(init)
        apply_init!(model, context, init_obj)
    end
    for (factor_id, child) in pairs(children(context))
        if (submodel = getspecificsubmodelinit(init, factor_id)) !== nothing
            apply_init!(model, child, getinitobjects(submodel))
        elseif (submodel = getgeneralsubmodelinit(init, fform(factor_id))) !== nothing
            apply_init!(model, child, getinitobjects(submodel))
        else
            apply_init!(model, child, default_init(fform(factor_id)))
        end
    end
end

function apply_init!(model::Model, context::Context, init::InitObject{S, T} where {S <: InitDescriptor, T})
    nodes = unroll(context[getvardescriptor(getvardescriptor(init))])
    apply_init!(model, context, init, nodes)
end

apply_init!(model::Model, context::Context, init::InitObject{S, T} where {S <: InitDescriptor, T}, node::NodeLabel) = save_init!(model, node, init)

function apply_init!(model::Model, context::Context, init::InitObject{S, T} where {S <: InitDescriptor, T}, nodes::AbstractArray{NodeLabel})
    for node in nodes
        save_init!(model, node, init)
    end
end

function apply_init!(model::Model, context::Context, init::InitObject{S, T}, nodes::AbstractArray{NodeLabel}) where {S <: InitDescriptor, T <: AbstractArray}
    for (node, marginal) in zip(nodes, getinitinfo(init))
        save_init!(model, node, InitObject(getvardescriptor(init), marginal))
    end
end

const InitMsgExtraKey = GraphPPL.NodeDataExtraKey{:init_msg, Any}()
const InitMarExtraKey = GraphPPL.NodeDataExtraKey{:init_mar, Any}()

save_init!(model::Model, node::NodeLabel, init::InitObject{S, T}) where {S <: InitDescriptor{InitMessage}, T} = save_init!(model, node, init, InitMsgExtraKey)
save_init!(model::Model, node::NodeLabel, init::InitObject{S, T}) where {S <: InitDescriptor{InitMarginal}, T} = save_init!(model, node, init, InitMarExtraKey)

function save_init!(model::Model, node::NodeLabel, init::InitObject{S, T} where {S, T}, key)
    nodedata = model[node]
    if !hasextra(nodedata, key)
        setextra!(nodedata, key, getinitinfo(init))
    end
end

struct NoInit end

"""
    MetaPlugin(init)

A plugin that adds a init information to the factor nodes of the model.
"""
struct InitializationPlugin{I}
    initialization::I
end

InitializationPlugin() = InitializationPlugin(NoInit())
InitializationPlugin(::Nothing) = InitializationPlugin(NoInit())

GraphPPL.plugin_type(::InitializationPlugin) = GraphPPL.VariableNodePlugin()

GraphPPL.preprocess_plugin(plugin::InitializationPlugin, model::Model, context::Context, label::NodeLabel, nodedata::GraphPPL.NodeData, options::GraphPPL.NodeCreationOptions) = label,
nodedata

function GraphPPL.postprocess_plugin(plugin::InitializationPlugin{NoInit}, model::Model)
    apply_init!(model, default_init(GraphPPL.fform(GraphPPL.getcontext(model))))
    return nothing
end

function GraphPPL.postprocess_plugin(plugin::InitializationPlugin, model::Model)
    apply_init!(model, plugin.initialization)
    return nothing
end

check_for_returns_init = (x) -> GraphPPL.check_for_returns(x; tag = "init")

function check_for_trailing_commas(e::Expr)
    if @capture(e, (a_ = (b_, c_) = d_))
        error("Trailing comma in init specification detected, please place every init statement on a new line without trailing commas.")
    end
    return e
end

function add_init_construction(e::Expr)
    if @capture(e, (function m_name_(m_args__; m_kwargs__)
        c_body_
    end) | (function m_name_(m_args__)
        c_body_
    end))
        m_kwargs = m_kwargs === nothing ? [] : m_kwargs
        return quote
            function $m_name($(m_args...); $(m_kwargs...))
                __init__ = RxInfer.InitSpecification()
                $c_body
                return __init__
            end
        end
    else
        return quote
            let __init__ = RxInfer.InitSpecification()
                $e
                __init__
            end
        end
    end
end

function create_submodel_init(e::Expr)
    if @capture(e, (
        for init in submodel_
            body__
        end
    ))
        if @capture(submodel, (name_, index_))
            submodel_constructor = :(RxInfer.SpecificSubModelInit(RxInfer.GraphPPL.FactorID($name, $index)))
        else
            submodel_constructor = :(RxInfer.GeneralSubModelInit($submodel))
        end
        return quote
            let __outer_init__ = __init__
                let __init__ = $submodel_constructor
                    $(body...)
                    push!(__outer_init__, __init__)
                end
            end
        end
    else
        return e
    end
end

"""
    convert_init_variables(e::Expr)

Converts all variable references on the left hand side of a init specification to IndexedVariable calls.

# Arguments
- `e::Expr`: The expression to convert.

# Returns
- `Expr`: The resulting expression with all variable references converted to IndexedVariable calls.

# Examples
"""

function convert_init_variables(e::Expr)
    if @capture(e, (fform_(var_) = init_obj_))
        var = GraphPPL.__convert_to_indexed_statement(var)
        return quote
            $fform($(var)) = $init_obj
        end
    end
    return e
end

what_walk(::typeof(convert_init_variables)) = walk_until_occurrence(:(lhs_ -> rhs_))

"""
    convert_init_node_aliases(e::Expr)

Converts user written initialization expressions to backend-aware node aliases (if defined),
with interface_aliases. So `@initialization` behaves consistently with `@model` definitons.
"""
function convert_init_node_aliases(e::Expr)
    if @capture(e, f_(args__))
        # created backend once for model (outside of function) -> better ideas?
        backend = INIT_BACKEND
        # Skip if f not a symbol
        if !(f isa Symbol)
            return e
        end
        # Skip if symbol cannot be resolved at macro time
        fform_type = try
            Base.eval(@__MODULE__, f) # Evaluates in RxInfer module context
        catch
            return e
        end
        # Skip if not a Distribution subtype (e.g. vague() -> will throw error for <: comparison)
        if !(fform_type isa Type)
            return e
        end
        # Skip if not with specific declared factor_alias functions in model/graphppl.jl
        try
            # get all declared methods for factor_alias
            # ml = methods(GraphPPL.factor_alias)
            ml = methods(GraphPPL.factor_alias, (typeof(backend), Type{fform_type}, Any))
            # Check whether any method's *concrete* 2nd signature parameter is exactly Type{fform_type}
            has_factor_alias = any(m -> begin
                sig = Base.unwrap_unionall(m.sig) # remove any UnionAll / where-quantifiers
                # sig is something like Tuple{factor_alias, ReactiveMPGraphPPLBackend, Type{Normal}, Any}
                sig.parameters[3] === Type{fform_type} # exact match to Type{<:fform_type}
            end, ml.ms)
            # Exit and return original expression if no factor alias is available
            if !has_factor_alias
                return e
            end
        catch err
            return e
        end
        
        # split keyword (NamedTuple) vs positional (Tuple) user calls 
        all_kwargs = all(arg -> arg isa Expr && arg.head == :kw, args)
        try
            if all_kwargs
                # keywords path: build NamedTuple of args, e.g. (mean=0.0, variance=1.0)
                keys_tuple = Tuple([arg.args[1] for arg in args])
                vals = Tuple([arg.args[2] for arg in args])
                rhs_nt = NamedTuple{keys_tuple}(vals)
               
                # Apply interface aliasing and map to canonical backend form (as in make_node!)
                # e.g. (mean=0.0, var=1.0) → (µ=0.0, w=1.0)
                aliased_nt = convert(
                    NamedTuple, 
                    GraphPPL.interface_aliases(GraphPPL.interface_aliases(backend, fform_type), GraphPPL.StaticInterfaces(keys(rhs_nt))),
                    values(rhs_nt))
                # Replace distribution constructor with aliased version, e.g.
                # Normal(mean, variance) → NormalMeanVariance(...)
                aliased_fform = GraphPPL.factor_alias(backend, fform_type, GraphPPL.StaticInterfaces(keys(aliased_nt)))
                # change aliased_fform from DataType to Expr - for passing in Expression tree
                # Expression trees should contain symbols or other expressions — not runtime type objects (like aliased_fform). 
                mod_sym = Symbol(string(parentmodule(aliased_fform))) # e.g. :ExponentialFamily
                name_sym = Symbol(string(nameof(aliased_fform))) # :NormalMeanVariance
                qualified_name = Expr(:., mod_sym, QuoteNode(name_sym)) # e.g. :ExponentialFamily.NormalMeanVariance
                return Expr(:call, qualified_name, map(identity, values(aliased_nt))...)
            else
                # positional path: use GraphPPL.default_parametrization e.g.
                # Normal(0.0, 1.0) → error: "Normal" needs kwargs
                nodetype = GraphPPL.NodeType(backend, fform_type)
                rhs_tuple = tuple((arg isa Expr && arg.head == :kw ? arg.args[2] : arg) for arg in args...)
                rhs_named = GraphPPL.default_parametrization(backend, nodetype, fform_type, rhs_tuple)
                # Resolve to backend’s aliased factor form (e.g. throw error if Normal is called without kwargs)
                aliased_fform = GraphPPL.factor_alias(backend, fform_type, GraphPPL.StaticInterfaces(keys(rhs_named)))
                return Expr(:call, aliased_fform, map(identity, values(rhs_named))...)
            end
        catch err
            rethrow(err)
        end
    end
    return e
end

what_walk(::typeof(convert_init_node_aliases)) = postwalk
const INIT_BACKEND = GraphPPL.instantiate(ReactiveMPGraphPPLBackend)

"""
    convert_init_object(e::Expr)

Converts a variable init or a factor init call on the left hand side of a init specification to a `GraphPPL.MetaObject`.

# Arguments
- `e::Expr`: The expression to convert.

# Returns
- `Expr`: The resulting expression with the variable reference or factor function call converted to a `GraphPPL.MetaObject`.

# Examples
"""
function convert_init_object(e::Expr)
    if @capture(e, (fform_(var_) = init_obj_))
        form = nothing
        if fform == :q
            form = :(RxInfer.InitMarginal())
        elseif fform == :μ
            form = :(RxInfer.InitMessage())
        end
        return quote
            push!(__init__, RxInfer.InitObject(RxInfer.InitDescriptor($form, $var), $init_obj))
        end
    else
        return e
    end
end

function init_macro_interior(init_body::Expr)
    init_body = GraphPPL.apply_pipeline(init_body, check_for_trailing_commas)
    init_body = GraphPPL.apply_pipeline(init_body, (x) -> check_for_returns_init(x))
    init_body = add_init_construction(init_body)
    init_body = GraphPPL.apply_pipeline(init_body, create_submodel_init)
    init_body = GraphPPL.apply_pipeline(init_body, convert_init_variables)
    init_body = GraphPPL.apply_pipeline(init_body, convert_init_node_aliases)
    init_body = GraphPPL.apply_pipeline(init_body, convert_init_object)
    return init_body
end

"""
    @initialization

Macro for specifying the initialization state of a model. Accepts either a function or a block of code.
Allows the specification of initial messages and marginals that can be applied to a model in the `infer` function.
"""
macro initialization(init_body)
    return esc(RxInfer.init_macro_interior(init_body))
end

const EmptyInit = @initialization begin end
