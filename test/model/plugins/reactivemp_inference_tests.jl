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
        (diagnostics = diagnostics, context = (rng = rng,), rulefallback = fallback),
    )

    @test RxInfer.getdiagnostics(options) === diagnostics
    @test RxInfer.getcontext(options) === (rng = rng,)
    @test RxInfer.getrulefallback(options) === fallback
    @test RxInfer.getrulefallback(RxInfer.setrulefallback(options, nothing)) === nothing
    @test RxInfer.getcontext(RxInfer.setcontext(options, nothing)) === nothing

    # A `NamedTuple` of the audits to switch on builds the `EngineDiagnostics`; the others stay off.
    shorthand = convert(ReactiveMPInferenceOptions, (diagnostics = (check_everything_pure = true, checked_buffers = true),))
    @test RxInfer.getdiagnostics(shorthand) === ReactiveMP.EngineDiagnostics(check_everything_pure = true, checked_buffers = true)
    @test_throws "Unknown diagnostics option: check_everything" convert(ReactiveMPInferenceOptions, (diagnostics = (check_everything = true,),))
    @test_throws "check_everything_inplace" convert(ReactiveMPInferenceOptions, (diagnostics = (check_everything = true,),))

    bad_nt = (blahblah = 1,)

    @test_throws "Unknown model inference options: blahblah" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
    @test_throws "Available options are" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
end

@testitem "A group of one member is the member `(name, 1)`" begin
    using RxInfer, DiscreteTransitionMessagePassingRules

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

    result = infer(model = one_control(), data = (y = [0.0, 0.0, 1.0], u = [0.0, 1.0]))
    joint = [p[i] * B[j, i, 2] * A[3, j] for j in 1:3, i in 1:3]
    @test probvec(result.posteriors[:s]) ≈ vec(sum(joint; dims = 2)) ./ sum(joint)
    @test probvec(result.posteriors[:x]) ≈ vec(sum(joint; dims = 1)) ./ sum(joint)
end
