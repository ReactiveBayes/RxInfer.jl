module ReactantExt

using RxInfer, Reactant

# Why this extension exists. Reactant compiles a function by tracing it: it runs the function once,
# rewriting every call inside it to Reactant's overlays. A call to `infer` compiled with Reactant
# therefore builds and runs the whole model under tracing. Since Reactant 0.2.292
# (EnzymeAD/Reactant.jl#3378), the overlays of `all`, `any` and `map` decide whether to trace by
# walking every field of every object in their arguments. The model's variables hold Rocket's linked
# lists, whose placeholder nodes have undefined fields and whose nodes point at each other, so the walk
# throws an `UndefRefError` or never ends, and compiling a model of more than one step fails.
# ReactiveMP's own extension marks the engine's graph building; the functions below are RxInfer's.
#
# They collect the model's variables, their streams and the results, and never compute with a
# variable's value, so they need no overlay and run as plain Julia under tracing. The rules, which do
# compute with the values, are not marked and stay traced. `getvariable` finds the engine's variables
# of the model's nodes, `getvarref` the references of the model's variables, `isanonymous` tells
# whether a variable has a name, `obtain_marginal` and `obtain_prediction` collect the streams of
# posteriors and predictions, and `inference_postprocess_or_missing` unpacks the results. The marks can go once Reactant's walk
# tolerates undefined fields and cycles (EnzymeAD/Reactant.jl#3506).
#
# `with_option` sets one field of the inference options and reads their `fieldnames`. On Julia 1.10,
# Reactant's rewrite of `fieldnames` on a `UnionAll` calls itself until the stack overflows, so it is
# marked too; it never sees a traced value.
#
# Grouped interfaces also resolve and cache their dependency positions during activation.
# Dependency resolution and the position cache only inspect graph structure, including Rocket
# observables. Their `count`/`map`/`any` operations otherwise walk undefined linked-list fields.
# Mark the helpers themselves: `planned_dependencies` and `resolve_dependencies` can be inlined
# before Reactant rewrites their caller. Joint clusters (such as addition inputs) also call
# `input_label` directly during activation. It only labels interfaces and counts group members.
# Message evaluation remains outside these helpers.
#
# The marks go into a set in Reactant, which precompilation does not keep, so they are made when the
# extension loads.
function __init__()
    Reactant.@skip_rewrite_func RxInfer.getvariable
    Reactant.@skip_rewrite_func RxInfer.getvarref
    Reactant.@skip_rewrite_func RxInfer.isanonymous
    Reactant.@skip_rewrite_func RxInfer.obtain_marginal
    Reactant.@skip_rewrite_func RxInfer.obtain_prediction
    Reactant.@skip_rewrite_func RxInfer.inference_postprocess_or_missing
    Reactant.@skip_rewrite_func RxInfer.with_option
    Reactant.@skip_rewrite_func RxInfer.ReactiveMP.group_length
    return nothing
end

end
