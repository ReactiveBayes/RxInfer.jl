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
    @test RxInfer.getrng(options) === nothing

    rng = StableRNG(42)
    diagnostics = ReactiveMP.EngineDiagnostics(checked_buffers = true)
    options = convert(ReactiveMPInferenceOptions, (diagnostics = diagnostics, rng = rng))

    @test RxInfer.getdiagnostics(options) === diagnostics
    @test RxInfer.getrng(options) === rng

    @test_throws "The `rulefallback` option is gone" convert(
        ReactiveMPInferenceOptions, (rulefallback = (args...) -> nothing,)
    )

    bad_nt = (blahblah = 1,)

    @test_throws "Unknown model inference options: blahblah" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
    @test_throws "Available options are" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
end
