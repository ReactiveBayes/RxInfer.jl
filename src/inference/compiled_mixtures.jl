# Grouped mixture interfaces use the existing rules and scoring math. They
# cannot use the ordinary factorization layout, which assumes unique names.
function lower_compiled_factor!(
    builder, fform::Type{<:Union{NormalMixture, GammaMixture}}, label, node
)
    neighbors = GraphPPL.neighbors(getproperties(node))
    names = Tuple(GraphPPL.getname(edge) for (_, edge, _) in neighbors)
    normal = fform <: NormalMixture
    first_name, second_name = normal ? (:m, :p) : (:a, :b)
    out = findfirst(==(:out), names)
    switch = findfirst(==(:switch), names)
    firsts = findall(==(first_name), names)
    seconds = findall(==(second_name), names)
    N = length(firsts)
    N >= 2 && N == length(seconds) || throw(
        ArgumentError(
            "Mixture requires matching groups of at least two components"
        ),
    )
    constraints = getextra(
        node, GraphPPL.VariationalConstraintsFactorizationIndicesKey
    )
    all(cluster -> length(cluster) == 1, constraints) || throw(
        ArgumentError("Mixture factorization must be the naive mean-field")
    )
    policy = getextra(node, ReactiveMPExtraDependenciesKey, nothing)
    supported_policy = if normal
        ReactiveMP.NormalMixtureNodeFunctionalDependencies
    else
        ReactiveMP.GammaMixtureNodeFunctionalDependencies
    end
    policy === nothing ||
        policy isa supported_policy ||
        throw(
            UnsupportedCompiledFeature(
                "mixture dependency policy $(typeof(policy))"
            ),
        )
    getextra(node, ReactiveMPExtraStreamPostprocessorsKey, nothing) ===
    nothing || throw(
        UnsupportedCompiledFeature("per-node stream postprocessors at $label"),
    )
    _, outputs, qs = compiled_factor_slots!(builder, neighbors)
    program = builder.program
    concrete = normal ? NormalMixture{N} : GammaMixture{N}
    meta = ReactiveMP.collect_meta(
        concrete, getextra(node, GraphPPL.MetaExtraKey, nothing)
    )
    first_group = ReactiveMP.CompiledManyOf(Tuple(qs[i] for i in firsts))
    second_group = ReactiveMP.CompiledManyOf(Tuple(qs[i] for i in seconds))
    function message!(i, tag, qnames, bindings)
        iszero(outputs[i]) && return nothing
        mapping = ReactiveMP.MessageMapping(
            concrete,
            tag,
            Marginalisation(),
            nothing,
            Val(qnames),
            meta,
            getannotations(builder.options),
            nothing,
            getrulefallback(builder.options),
            compiled_numerical_callbacks(builder.options),
        )
        ReactiveMP.compiled_operation!(
            program,
            ReactiveMP.CompiledMessageKernel(mapping),
            outputs[i],
            (nothing, bindings),
        )
    end
    message!(
        out,
        Val(:out),
        (:switch, first_name, second_name),
        (qs[switch], first_group, second_group),
    )
    message!(
        switch,
        Val(:switch),
        (:out, first_name, second_name),
        (qs[out], first_group, second_group),
    )
    for k in 1:N
        message!(
            firsts[k],
            (Val(first_name), k),
            (:out, :switch, second_name),
            (qs[out], qs[switch], qs[seconds[k]]),
        )
        message!(
            seconds[k],
            (Val(second_name), k),
            (:out, :switch, first_name),
            (qs[out], qs[switch], qs[firsts[k]]),
        )
    end
    if builder.score_type !== nothing
        slot = ReactiveMP.compiled_slot!(program)
        kernel = CompiledStochasticScore(
            concrete,
            Val((:out, :switch, first_name, second_name)),
            meta,
            builder.score_type,
        )
        ReactiveMP.compiled_operation!(
            program,
            kernel,
            slot,
            (qs[out], qs[switch], first_group, second_group),
        )
        push!(builder.score_slots, slot)
    end
    return nothing
end

function lower_compiled_factor!(builder, fform::Type{<:Mixture}, label, node)
    builder.score_type === nothing || throw(
        UnsupportedCompiledFeature(
            "Mixture free energy (no reference scoring rule)"
        ),
    )
    neighbors = GraphPPL.neighbors(getproperties(node))
    names = Tuple(GraphPPL.getname(edge) for (_, edge, _) in neighbors)
    out, switch = findfirst(==(:out), names), findfirst(==(:switch), names)
    components = findall(==(:inputs), names)
    policy = getextra(node, ReactiveMPExtraDependenciesKey, nothing)
    hard_switch = policy isa ReactiveMP.RequireMarginalFunctionalDependencies
    policy === nothing ||
        hard_switch ||
        policy isa ReactiveMP.MixtureNodeFunctionalDependencies ||
        throw(
            UnsupportedCompiledFeature(
                "Mixture dependency policy $(typeof(policy))"
            ),
        )
    getextra(node, ReactiveMPExtraStreamPostprocessorsKey, nothing) ===
    nothing || throw(
        UnsupportedCompiledFeature("per-node stream postprocessors at $label"),
    )
    inputs, outputs, qs = compiled_factor_slots!(builder, neighbors)
    program = builder.program
    concrete = Mixture{length(components)}
    meta = ReactiveMP.collect_meta(
        concrete, getextra(node, GraphPPL.MetaExtraKey, nothing)
    )
    grouped = ReactiveMP.CompiledManyOf(Tuple(inputs[i] for i in components))
    function message!(
        i, tag, mnames, bindings, qnames = nothing, marginals = nothing
    )
        iszero(outputs[i]) && return nothing
        mapping = ReactiveMP.MessageMapping(
            concrete,
            tag,
            Marginalisation(),
            Val(mnames),
            qnames,
            meta,
            getannotations(builder.options),
            nothing,
            getrulefallback(builder.options),
            compiled_numerical_callbacks(builder.options),
        )
        ReactiveMP.compiled_operation!(
            program,
            ReactiveMP.CompiledMessageKernel(mapping),
            outputs[i],
            (bindings, marginals),
        )
    end
    if hard_switch
        message!(
            out,
            Val(:out),
            (:inputs,),
            (grouped,),
            Val((:switch,)),
            (qs[switch],),
        )
    else
        message!(out, Val(:out), (:switch, :inputs), (inputs[switch], grouped))
    end
    message!(switch, Val(:switch), (:out, :inputs), (inputs[out], grouped))
    for (k, i) in enumerate(components)
        if hard_switch
            message!(
                i,
                (Val(:inputs), k),
                (:out,),
                (inputs[out],),
                Val((:switch,)),
                (qs[switch],),
            )
        else
            message!(
                i,
                (Val(:inputs), k),
                (:out, :switch),
                (inputs[out], inputs[switch]),
            )
        end
    end
    return nothing
end
