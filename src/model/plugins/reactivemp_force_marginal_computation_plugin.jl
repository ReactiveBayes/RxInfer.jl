import ReactiveMP:
    getlocalclusters,
    get_stream_of_marginals,
    schedule_on,
    sdtype,
    getinterfaces,
    name,
    get_stream_of_inbound_messages,
    getinboundinterfaces,
    functionalform,
    Marginal,
    marginalrule,
    clustername

const ReactiveMPExtraMarginalStreamKey = GraphPPL.NodeDataExtraKey{
    :marginal_stream, Any
}()

"""
A plugin for GraphPPL graph engine that forces the computation of marginal distributions for every node in the graph.
"""
struct ReactiveMPForceMarginalComputationPlugin end

GraphPPL.plugin_type(::ReactiveMPForceMarginalComputationPlugin) =
    GraphPPL.FactorNodePlugin()

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
    factor_nodes(model) do _, node
        factornode = getextra(node, ReactiveMPExtraFactorNodeKey)
        metadata = getextra(node, GraphPPL.MetaExtraKey, nothing)
        subscription = create_marginals_stream(factornode, metadata)
        setextra!(node, ReactiveMPExtraMarginalStreamKey, subscription)
    end
    return nothing
end

function create_marginals_stream(node::ReactiveMP.AbstractFactorNode, meta)
    return create_marginals_stream(sdtype(node), node, meta)
end

function create_marginals_stream(
    ::Deterministic, node::ReactiveMP.AbstractFactorNode, meta
)
    fnstream = (interface) -> get_stream_of_inbound_messages(interface)
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
            on_complete = () -> nothing,
        ),
    )
end

function create_marginals_stream(
    ::Stochastic, node::ReactiveMP.AbstractFactorNode, meta
)
    fnstream = (localmarginal) -> get_stream_of_marginals(localmarginal)
    localmarginals = ReactiveMP.get_node_local_marginals(getlocalclusters(node))
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
            on_complete = () -> nothing,
        ),
    )
end
