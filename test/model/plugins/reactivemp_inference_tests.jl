@testitem "ReactiveMPInferenceOptions can be constructed" begin
    import RxInfer: ReactiveMPInferenceOptions
    import ReactiveMP: AbstractAnnotations

    struct MyStreamPostprocessor end
    struct MyAnotherStreamPostprocessor end
    struct MyAnnotations <: AbstractAnnotations end
    struct MyAnotherAnnotations <: AbstractAnnotations end

    options = ReactiveMPInferenceOptions(
        MyStreamPostprocessor(), MyAnnotations()
    )

    @test RxInfer.getpostprocessor(options) === MyStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnnotations(),)
    @test RxInfer.getdiagnostics(options) === ReactiveMP.EngineDiagnostics()
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setpostprocessor(options, MyAnotherStreamPostprocessor())

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnnotations(),)
    @test RxInfer.getdiagnostics(options) === ReactiveMP.EngineDiagnostics()
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setannotations(options, MyAnotherAnnotations())

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getdiagnostics(options) === ReactiveMP.EngineDiagnostics()
    @test RxInfer.getcallbacks(options) === nothing

    diagnostics = ReactiveMP.EngineDiagnostics(check_everything_pure = true)
    options = RxInfer.setdiagnostics(options, diagnostics)

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getdiagnostics(options) === diagnostics
    @test RxInfer.getcallbacks(options) === nothing

    callbacks = (args...) -> print(args...)
    options = RxInfer.setcallbacks(options, callbacks)

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getdiagnostics(options) === diagnostics
    @test RxInfer.getcallbacks(options) === callbacks
end

@testitem "ReactiveMPInferenceOptions can be converted from NamedTuple" begin
    import RxInfer: ReactiveMPInferenceOptions
    using StableRNGs

    struct MyStreamPostprocessorForNamedTuple end

    callbacks = (args...) -> nothing
    nt = (
        stream_postprocessors = MyStreamPostprocessorForNamedTuple(),
        callbacks = callbacks,
    )

    options = convert(ReactiveMPInferenceOptions, nt)

    @test RxInfer.getpostprocessor(options) ===
        MyStreamPostprocessorForNamedTuple()
    @test RxInfer.getcallbacks(options) === callbacks
    @test RxInfer.getdiagnostics(options) === ReactiveMP.EngineDiagnostics()
    @test RxInfer.getcontext(options) === nothing
    @test RxInfer.getrulefallback(options) === nothing

    rng = StableRNG(42)
    diagnostics = ReactiveMP.EngineDiagnostics(checked_buffers = true)
    fallback = NodeFunctionRuleFallback()
    options = convert(
        ReactiveMPInferenceOptions,
        (
            diagnostics = diagnostics,
            context = (rng = rng,),
            rulefallback = fallback,
        ),
    )

    @test RxInfer.getdiagnostics(options) === diagnostics
    @test RxInfer.getcontext(options) === (rng = rng,)
    @test RxInfer.getrulefallback(options) === fallback
    @test RxInfer.getrulefallback(RxInfer.setrulefallback(options, nothing)) ===
        nothing
    @test RxInfer.getcontext(RxInfer.setcontext(options, nothing)) === nothing

    # A `NamedTuple` of the audits to switch on builds the `EngineDiagnostics`; the others stay off.
    shorthand = convert(
        ReactiveMPInferenceOptions,
        (diagnostics = (check_everything_pure = true, checked_buffers = true),),
    )
    @test RxInfer.getdiagnostics(shorthand) === ReactiveMP.EngineDiagnostics(
        check_everything_pure = true, checked_buffers = true
    )
    @test_throws "Unknown diagnostics option: check_everything" convert(
        ReactiveMPInferenceOptions, (diagnostics = (check_everything = true,),)
    )
    @test_throws "check_everything_inplace" convert(
        ReactiveMPInferenceOptions, (diagnostics = (check_everything = true,),)
    )

    bad_nt = (blahblah = 1,)

    @test_throws "Unknown model inference options: blahblah" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
    @test_throws "Available options are" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
end

@testitem "A group of one member is the member `(name, 1)`" begin
    using RxInfer

    # `DiscreteTransition(x, B, u)` gives the group `T` one member, which GraphPPL does not index
    normalised(A) = A ./ sum(A; dims = 1)
    B = normalised(reshape(Float64.(1:18), 3, 3, 2))
    A = normalised([4.0 1.0 1.0; 1.0 3.0 1.0; 1.0 1.0 2.0])
    p = [0.2, 0.5, 0.3]

    @model function one_control(y, u)
        x ~ Categorical(p)
        s ~ DiscreteTransition(x, B, u)
        y ~ DiscreteTransition(s, A)
    end

    result = infer(
        model = one_control(), data = (y = [0.0, 0.0, 1.0], u = [0.0, 1.0])
    )
    joint = [p[i] * B[j, i, 2] * A[3, j] for j in 1:3, i in 1:3]
    @test probvec(result.posteriors[:s]) ≈
        vec(sum(joint; dims = 2)) ./ sum(joint)
    @test probvec(result.posteriors[:x]) ≈
        vec(sum(joint; dims = 1)) ./ sum(joint)
end

@testitem "An algorithm names variables of the model, resolved to the node's interfaces" begin
    using RxInfer
    import RxInfer: getmodel, node_algorithm
    import GraphPPL: factor_nodes, fform, getproperties

    struct Roles{C, P} <: DefaultAlgorithmExtension
        control::C
        parameter::P
    end
    RxInfer.resolve_variables(roles::Roles, interface) =
        Roles(interface(roles.control), interface(roles.parameter))

    normalised(A) = A ./ sum(A; dims = 1)
    A = normalised(reshape(Float64.(1:8), 2, 2, 2))
    B = normalised(reshape(Float64.(1:12), 3, 2, 2))

    # `u[t]` and `x[t]` are members of collections and arrive as their labels, `θ` and `y` as
    # references to them
    @model function planner(A, B, goal)
        θ ~ Categorical([0.4, 0.6])
        x[1] ~ Categorical([1.0, 0.0])
        for t in 1:2
            u[t] ~ Categorical([0.5, 0.5])
            x[t + 1] ~ DiscreteTransition(x[t], A, u[t]) where {
                algorithm = Roles(u[t], x[t]),
            }
        end
        y ~ DiscreteTransition(x[3], B, θ) where {algorithm = Roles(θ, y)}
        y ~ Categorical(goal)
    end

    # `Roles` adds no rules, so the model runs as it does without them, the free energy, which
    # reads the algorithm too, included
    @model function plain(A, B, goal)
        θ ~ Categorical([0.4, 0.6])
        x[1] ~ Categorical([1.0, 0.0])
        for t in 1:2
            u[t] ~ Categorical([0.5, 0.5])
            x[t + 1] ~ DiscreteTransition(x[t], A, u[t])
        end
        y ~ DiscreteTransition(x[3], B, θ)
        y ~ Categorical(goal)
    end

    data = (goal = [0.2, 0.3, 0.5],)
    result = infer(model = planner(A = A, B = B), data = data, free_energy = true)
    reference = infer(model = plain(A = A, B = B), data = data, free_energy = true)

    algorithms = []
    factor_nodes(getmodel(result.model)) do _, nodedata
        fform(getproperties(nodedata)) === DiscreteTransition &&
            push!(algorithms, node_algorithm(nodedata))
    end
    @test count(==(Roles((:T, 1), :in)), algorithms) == 2
    @test count(==(Roles((:T, 1), :out)), algorithms) == 1
    @test probvec(result.posteriors[:u][1]) ≈ probvec(reference.posteriors[:u][1])
    @test result.free_energy ≈ reference.free_energy
end

@testitem "An algorithm that names a variable the node is not connected to is an error" begin
    using RxInfer

    struct Held{V} <: DefaultAlgorithmExtension
        variable::V
    end
    RxInfer.resolve_variables(held::Held, interface) = Held(interface(held.variable))

    @model function unconnected(y, w)
        z ~ NormalMeanVariance(0.0, 1.0)
        w ~ NormalMeanVariance(z, 1.0)
        x ~ NormalMeanVariance(0.0, 1.0)
        y ~ NormalMeanVariance(x, 1.0) where {algorithm = Held(z)}
    end

    @test_throws "names the variable `z`, which the node is not connected to" infer(
        model = unconnected(), data = (y = 1.0, w = 0.0)
    )
end

@testitem "The inputs of a Delta node are the members of `in`" begin
    using RxInfer
    import RxInfer: getmodel, node_interface_key
    import GraphPPL: factor_nodes, fform, getproperties, getcontext

    f(a) = 2a
    g(a, b) = a + b

    @model function deltas(y)
        a ~ NormalMeanVariance(0.0, 1.0)
        b ~ NormalMeanVariance(0.0, 1.0)
        c := f(a)
        d := g(a, b)
        y ~ NormalMeanVariance(c + d, 1.0)
    end

    model = getmodel(RxInfer.create_model(deltas() | (y = 1.0,)))
    context = getcontext(model)
    keys = Dict()
    factor_nodes(model) do _, nodedata
        properties = getproperties(nodedata)
        fform(properties) === f &&
            (keys[f] = (node_interface_key(properties, context[:a]),))
        fform(properties) === g && (
            keys[g] = (
                node_interface_key(properties, context[:a]),
                node_interface_key(properties, context[:b]),
            )
        )
    end
    @test keys[f] == ((:in, 1),)
    @test keys[g] == ((:in, 1), (:in, 2))
end
