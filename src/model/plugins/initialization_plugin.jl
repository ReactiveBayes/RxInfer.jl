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
    convert_init_fform(init_obj::Expr)

Converts distribution constructor calls with (kw)args to use RxInfer.convert_fform.
Skips conversion for special functions like `vague()` and `huge()`.
"""
function convert_init_fform(e::Expr)
    if @capture(e, (fform_(var_) = init_obj_))
        # local helper: convert the RHS init_obj expression if needed
            function _convert_init_obj(init_obj)
                # Recursively unwrap nested block expressions
                while init_obj isa Expr && init_obj.head == :block
                    actual_content = filter(x -> !(x isa LineNumberNode), init_obj.args)
                    if length(actual_content) == 1
                        init_obj = actual_content[1]
                    else
                        break
                    end
                end
                # Filter for function call pattern func(args...) e.g. Normal(0,1)
                if init_obj isa Expr && @capture(init_obj, func_(allargs__))
                    # Skip if already converted (explicitly RxInfer.convert_fform call)
                    if (func isa Expr && @capture(func, (mod_.convert_fform)))
                        return init_obj
                    end
    
                    # Skip special functions
                    if func in (:vague, :huge, :tiny)
                        return init_obj
                    end
    
                    # For repeat() and similar: recursively convert arguments
                    if func == :repeat
                        converted_args = map(_convert_init_obj, allargs)
                        return Expr(:call, :repeat, converted_args...)
                    end
    
                    # Separate kwargs from positional args
                    kwargs = filter(x -> x isa Expr && x.head == :kw, allargs)
                    args = filter(x -> !(x isa Expr && x.head == :kw), allargs)
    
                    if isempty(args) && !isempty(kwargs)
                        # All kwargs -> create NamedTuple from kw pairs
                        names = Tuple(kw.args[1] for kw in kwargs)
                        vals  = Tuple(kw.args[2] for kw in kwargs)
                        nt_expr = Expr(:tuple, (Expr(:(=), name, val) for (name, val) in zip(names, vals))...)
                        return :(RxInfer.convert_fform($func, (;$nt_expr...)))
                    elseif !isempty(allargs)
                        # Positional or mixed -> forward all args
                        return :(RxInfer.convert_fform($func, $(allargs...)))
                    end
            end

            # Handle vector literals [...]
            if init_obj isa Expr && init_obj.head == :vect
                converted_elements = map(_convert_init_obj, init_obj.args)
                return Expr(:vect, converted_elements...)
            end

            # nothing to convert
            return init_obj
        end
            
        converted_init = _convert_init_obj(init_obj)
        return quote
            $fform($(var)) = $converted_init
        end
    end
    return e
end

what_walk(::typeof(convert_init_fform)) = walk_until_occurrence(:(lhs_ -> rhs_))
const INIT_BACKEND = GraphPPL.instantiate(ReactiveMPGraphPPLBackend)

# convert_fform Kwargs version - all arguments are keyword arguments
function convert_fform(
    fform::F,
    kwargs::NamedTuple;
    backend = INIT_BACKEND
) where {F}
    try
        # Get interface aliases for this factor form
        aliases = GraphPPL.interface_aliases(backend, fform)
        # Convert user-provided kwargs to canonical interface names
        aliased_kwargs = GraphPPL.convert(
            NamedTuple, 
            GraphPPL.interface_aliases(aliases, GraphPPL.StaticInterfaces(keys(kwargs))), 
            values(kwargs)
        )
        # Resolve to backend's aliased factor form (e.g., Normal -> NormalMeanVariance)
        aliased_fform = GraphPPL.factor_alias(
            backend, 
            fform, 
            GraphPPL.StaticInterfaces(keys(aliased_kwargs))
        )
        # Construct the aliased factor with canonical kwargs
        return GraphPPL.__evaluate_fform(aliased_fform, values(aliased_kwargs))
        
    catch err
        error("Initialization macro returns error for $fform with kwargs $kwargs: $err")
    end
end

# Args version - positional or mixed arguments
function convert_fform(
    fform::F,
    args...;
    backend = INIT_BACKEND
) where {F}
    try
        nodetype = GraphPPL.NodeType(backend, fform)
        
        # Extract values from potential :kw expressions
        rhs_tuple = Tuple((arg isa Expr && arg.head == :kw ? arg.args[2] : arg) for arg in args)
        
        # Get default parametrization (may error if kwargs required)
        rhs_named = GraphPPL.default_parametrization(backend, nodetype, fform, rhs_tuple)
        
        # Resolve to backend's aliased factor form
        aliased_fform = GraphPPL.factor_alias(
            backend, 
            fform, 
            GraphPPL.StaticInterfaces(keys(rhs_named))
        )
        # Construct with values in order
        rhs_values = values(rhs_named)
        if length(rhs_values) == 1 && fform == PointMass
            return aliased_fform(rhs_values[1]...)
        else
            return aliased_fform(rhs_values...)
        end  
    catch err
        error("Initialization macro returns error for $fform with args $args: $err")
    end
end

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
    init_body = GraphPPL.apply_pipeline(init_body, convert_init_fform)
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
