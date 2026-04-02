@testitem "ReactiveMPInferenceOptions can be constructed" begin
    import RxInfer: ReactiveMPInferenceOptions
    import ReactiveMP: AbstractAddon

    struct MyScheduler end
    struct MyAnotherScheduler end
    struct MyAddons <: AbstractAddon end
    struct MyAnotherAddons <: AbstractAddon end

    options = ReactiveMPInferenceOptions(MyScheduler(), MyAddons())

    @test RxInfer.getscheduler(options) === MyScheduler()
    @test RxInfer.getaddons(options) === (MyAddons(),)
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setscheduler(options, MyAnotherScheduler())

    @test RxInfer.getscheduler(options) === MyAnotherScheduler()
    @test RxInfer.getaddons(options) === (MyAddons(),)
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    options = RxInfer.setaddons(options, MyAnotherAddons())

    @test RxInfer.getscheduler(options) === MyAnotherScheduler()
    @test RxInfer.getaddons(options) === (MyAnotherAddons(),)
    @test RxInfer.getrulefallback(options) === nothing
    @test RxInfer.getcallbacks(options) === nothing

    rulefallback = (args...) -> print(args)
    options = RxInfer.setrulefallback(options, rulefallback)

    @test RxInfer.getscheduler(options) === MyAnotherScheduler()
    @test RxInfer.getaddons(options) === (MyAnotherAddons(),)
    @test RxInfer.getrulefallback(options) === rulefallback
    @test RxInfer.getcallbacks(options) === nothing

    callbacks = (args...) -> print(args...)
    options = RxInfer.setcallbacks(options, callbacks)

    @test RxInfer.getscheduler(options) === MyAnotherScheduler()
    @test RxInfer.getaddons(options) === (MyAnotherAddons(),)
    @test RxInfer.getrulefallback(options) === rulefallback
    @test RxInfer.getcallbacks(options) === callbacks
end

@testitem "ReactiveMPInferenceOptions can be converted from NamedTuple" begin
    import RxInfer: ReactiveMPInferenceOptions

    struct MySchedulerForNamedTuple end

    callbacks = (args...) -> nothing
    nt = (scheduler = MySchedulerForNamedTuple(), callbacks = callbacks)

    options = convert(ReactiveMPInferenceOptions, nt)

    @test RxInfer.getscheduler(options) === MySchedulerForNamedTuple()
    @test RxInfer.getcallbacks(options) === callbacks

    bad_nt = (blahblah = 1,)

    @test_throws "Unknown model inference options: blahblah" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
    @test_throws "Available options are" convert(
        ReactiveMPInferenceOptions, bad_nt
    )
end
