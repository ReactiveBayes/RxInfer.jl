import ReactiveMP: getlocalclusters, get_stream_of_marginals

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
        subscription = create_marginals_stream(factornode)
        setextra!(node, ReactiveMPExtraMarginalStreamKey, subscription)
    end
    return nothing
end

# Subscribing to a node's local marginals computes them: a joint by the node's marginal rule, and
# a deterministic node's always has one, over its inputs.
function create_marginals_stream(node::ReactiveMP.AbstractFactorNode)
    localmarginals = ReactiveMP.get_node_local_marginals(getlocalclusters(node))
    stream = combineLatest(map(get_stream_of_marginals, localmarginals), PushNew())
    return subscribe!(
        stream |> map(Nothing, (_) -> nothing),
        lambda(
            Nothing;
            on_next = (d) -> nothing,
            on_error = (e) -> error(e),
            on_complete = () -> nothing,
        ),
    )
end
