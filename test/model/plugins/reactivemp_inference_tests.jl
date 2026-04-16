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
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setpostprocessor(options, MyAnotherStreamPostprocessor())

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnnotations(),)
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setannotations(options, MyAnotherAnnotations())

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    rulefallback = (args...) -> print(args)
    options = RxInfer.setrulefallback(options, rulefallback)

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getrulefallback(options) === rulefallback
    @test RxInfer.getcallbacks(options) === nothing

    callbacks = (args...) -> print(args...)
    options = RxInfer.setcallbacks(options, callbacks)

    @test RxInfer.getpostprocessor(options) === MyAnotherStreamPostprocessor()
    @test RxInfer.getannotations(options) === (MyAnotherAnnotations(),)
    @test RxInfer.getrulefallback(options) === rulefallback
    @test RxInfer.getcallbacks(options) === callbacks
end

@testitem "ReactiveMPInferenceOptions can be converted from NamedTuple" begin
    import RxInfer: ReactiveMPInferenceOptions

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

    bad_nt = (blahblah = 1,)

    @test_throws "Unknown model inference options: blahblah" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
    @test_throws "Available options are" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
end
