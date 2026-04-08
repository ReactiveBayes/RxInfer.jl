import ReactiveMP:
    getlocalclusters,
    apply_skip_filter,
    sdtype,
    getinterfaces,
    name,
    messagein,
    getinboundinterfaces,
    functionalform,
    Marginal,
    marginalrule,
    clustername

const MarginalComputationDefaultSkipStrategy = IncludeAll()

const ReactiveMPExtraMarginalStreamKey = GraphPPL.NodeDataExtraKey{
    :marginal_stream, Any
}()

struct MarginalComputationOptions{M}
    skip_strategy::M
end

MarginalComputationOptions() = MarginalComputationOptions(
    MarginalComputationDefaultSkipStrategy
)

get_skip_strategy(options::MarginalComputationOptions) = options.skip_strategy

"""
A plugin for GraphPPL graph engine that forces the computation of marginal distributions for every node in the graph.
"""
struct ReactiveMPForceMarginalComputationPlugin{O}
    options::O
end

getoptions(plugin::ReactiveMPForceMarginalComputationPlugin) = plugin.options

GraphPPL.plugin_type(::ReactiveMPForceMarginalComputationPlugin) = GraphPPL.FactorNodePlugin()

function GraphPPL.preprocess_plugin(
    ::ReactiveMPForceMarginalComputationPlugin,
    ::Model,
    ::Context,
    label::NodeLabel,
    nodedata::NodeData,
    ::NodeCreationOptions,
)
    return label, nodedata
end

function GraphPPL.postprocess_plugin(
    plugin::ReactiveMPForceMarginalComputationPlugin, model::Model
)
    return postprocess_plugin(plugin, getoptions(plugin), model)
end

function GraphPPL.postprocess_plugin(
    plugin::ReactiveMPForceMarginalComputationPlugin,
    options::MarginalComputationOptions,
    model::Model,
)
    skip_strategy = get_skip_strategy(options)

    factor_nodes(model) do _, node
        factornode = getextra(node, ReactiveMPExtraFactorNodeKey)
        metadata = getextra(node, GraphPPL.MetaExtraKey, nothing)
        subscription = create_marginals_stream(
            factornode, metadata, skip_strategy
        )
        setextra!(node, ReactiveMPExtraMarginalStreamKey, subscription)
    end
    return nothing
end

function create_marginals_stream(
    node::ReactiveMP.AbstractFactorNode, meta, skip_strategy
)
    return create_marginals_stream(
        sdtype(node), node, meta, skip_strategy
    )
end

function create_marginals_stream(
    ::Deterministic,
    node::ReactiveMP.AbstractFactorNode,
    meta,
    skip_strategy,
)
    fnstream = let skip_strategy = skip_strategy
        (interface) -> apply_skip_filter(messagein(interface), skip_strategy)
    end

    tinterfaces = Tuple(getinterfaces(node))
    stream = combineLatest(map(fnstream, tinterfaces), PushNew())

    vtag       = Val{clustername(getinboundinterfaces(node))}()
    msgs_names = Val{map(name, tinterfaces)}()

    mapping =
        let fform = functionalform(node),
            vtag = vtag,
            msgs_names = msgs_names,
            node = node

            (messages) -> begin
                # We do not really care about (is_clamped, is_initial) at this stage, so it can be (false, false)
                marginal = Marginal(
                    marginalrule(
                        fform,
                        vtag,
                        msgs_names,
                        messages,
                        nothing,
                        nothing,
                        meta,
                        node,
                    ),
                    false,
                    false,
                )
                return nothing
            end
        end

    s = stream |> map(Nothing, mapping)
    return subscribe!(
        s,
        lambda(
            Nothing;
            on_next = (d) -> nothing,
            on_error = (e) -> error(e),
            on_complete = () -> println("Completed"),
        ),
    )
end

function create_marginals_stream(
    ::Stochastic,
    node::ReactiveMP.AbstractFactorNode,
    meta,
    skip_strategy,
)
    fnstream = let skip_strategy = skip_strategy
        (localmarginal) -> apply_skip_filter(getmarginal(localmarginal), skip_strategy)
    end

    localmarginals = getmarginals(getlocalclusters(node))
    stream = combineLatest(map(fnstream, localmarginals), PushNew())

    mapping =
        let fform = functionalform(node),
            marginal_names = Val{Tuple(map(name, localmarginals))}()

            (marginals) -> begin
                return nothing
            end
        end
    s = stream |> map(Nothing, mapping)
    return subscribe!(
        s,
        lambda(
            Nothing;
            on_next = (d) -> nothing,
            on_error = (e) -> error(e),
            on_complete = () -> println("Completed"),
        ),
    )
end
