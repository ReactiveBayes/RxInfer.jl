
## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

using Distributed

const worker_io_lock = ReentrantLock()
const worker_ios     = Dict()

worker_io(ident) = get!(() -> IOBuffer(), worker_ios, string(ident))

# Dynamically overwrite default worker's `print` function for better control over stdout
Distributed.redirect_worker_output(ident, stream) = begin
    task = @async while !eof(stream)
        line = readline(stream)
        lock(worker_io_lock) do
            io = worker_io(ident)
            write(io, line, "\n")
        end
    end
    @static if VERSION >= v"1.7"
        Base.errormonitor(task)
    end
end

# This function prints `worker's` standard output into the global standard output
function flush_workerio(ident)
    lock(worker_io_lock) do
        wio = worker_io(ident)
        str = String(take!(wio))
        println(stdout, str)
        flush(stdout)
    end
end

# Unregistered GraphPPL and ReactiveMP, do not commit this two lines, but use them to test ReactiveMP locally
ENV["JULIA_PKG_USE_CLI_GIT"] = true
import Pkg; Pkg.rm("GraphPPL"); Pkg.add(Pkg.PackageSpec(name = "GraphPPL", rev = "develop-3.0"));
import Pkg; Pkg.rm("ReactiveMP"); Pkg.add(Pkg.PackageSpec(name = "ReactiveMP", rev = "develop-3.0"));

# DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer, Distributions); recursive=true)

# Example usage of a reduced testset
# julia --project --color=yes -e 'import Pkg; Pkg.test(test_args = [ "distributions:normal_mean_variance" ])'

addprocs(min(Sys.CPU_THREADS, 4))

@everywhere using Test, Documenter, RxInfer
@everywhere using TestSetExtensions

import Base: wait

mutable struct TestRunner
    enabled_tests
    found_tests
    test_tasks
    workerpool
    jobschannel
    exschannel
    iochannel

    function TestRunner(ARGS)
        enabled_tests = lowercase.(ARGS)
        found_tests = Dict(map(test -> test => false, enabled_tests))
        test_tasks = []
        jobschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for jobs
        exschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for exceptions
        iochannel = RemoteChannel(() -> Channel(0), myid())
        @async begin
            while isopen(iochannel)
                ident = take!(iochannel)
                flush_workerio(ident)
            end
        end
        return new(enabled_tests, found_tests, test_tasks, 2:nprocs(), jobschannel, exschannel, iochannel)
    end
end

function Base.run(testrunner::TestRunner)
    println("") # New line for 'better' alignment of the `testrunner` results

    foreach(testrunner.workerpool) do worker
        # For each worker we create a `nothing` token in the `jobschannel`
        # This token indicates that there are no other jobs left
        put!(testrunner.jobschannel, nothing)
        # We create a remote call for another Julia process to execute our test with `include(filename)`
        task = remotecall(
            worker,
            testrunner.jobschannel,
            testrunner.exschannel,
            testrunner.iochannel
        ) do jobschannel, exschannel, iochannel
            finish = false
            while !finish
                # Each worker takes jobs sequentially from the shared jobs pool 
                job_filename = take!(jobschannel)
                if isnothing(job_filename) # At the end there are should be only `emptyjobs`, in which case the worker finishes its tasks
                    finish = true
                else # Otherwise we assume that the `job` contains the valid `filename` and execute test
                    try # Here we can easily get the `LoadError` if some tests are failing
                        include(job_filename)
                    catch iexception
                        put!(exschannel, iexception)
                    end
                    # After the work is done we put the worker's `id` into `iochannel` (this triggers test info printing)
                    put!(iochannel, myid())
                end
            end
            return nothing
        end
        # We save the created task for later syncronization
        push!(testrunner.test_tasks, task)
    end

    # For each remotelly called task we `fetch` its result or save an exception
    foreach(fetch, testrunner.test_tasks)

    # If exception are not empty we notify the user and force-fail
    if isready(testrunner.exschannel)
        println(stderr, "Tests have failed with the following exceptions: ")
        while isready(testrunner.exschannel)
            exception = take!(testrunner.exschannel)
            showerror(stderr, exception)
            println(stderr, "\n", "="^80)
        end
        exit(-1)
    end

    close(testrunner.iochannel)
    close(testrunner.exschannel)
    close(testrunner.jobschannel)

    # At the very last stage we check that there are no "missing" tests, 
    # aka tests that have been specified in the `enabled_tests`, 
    # but for which the corresponding `filename` does not exist in the `test/` folder
    notfound_tests = filter(v -> v[2] === false, testrunner.found_tests)
    if !isempty(notfound_tests)
        println(stderr, "There are missing tests, double check correct spelling/path for the following entries:")
        foreach(keys(notfound_tests)) do key
            println(stderr, " - ", key)
        end
        exit(-1)
    end
end

const testrunner = TestRunner(lowercase.(ARGS))

println("`TestRunner` has been created. The number of available procs is $(nprocs()).")

@everywhere workerlocal_lock = ReentrantLock()

function addtests(testrunner::TestRunner, filename)
    # First we transform filename into `key` and check if we have this entry in the `enabled_tests` (if `enabled_tests` is not empty)
    key = filename_to_key(filename)
    if isempty(testrunner.enabled_tests) || key in testrunner.enabled_tests
        # If `enabled_tests` is not empty we mark the corresponding key with the `true` value to indicate that we found the corresponding `file` in the `/test` folder
        if !isempty(testrunner.enabled_tests)
            setindex!(testrunner.found_tests, true, key) # Mark that test has been found
        end
        # At this stage we simply put the `filename` into the `jobschannel` that will be processed later (see the `execute` function)
        put!(testrunner.jobschannel, filename)
    end
end

function key_to_filename(key)
    splitted = split(key, ":")
    return if length(splitted) === 1
        string("test_", first(splitted), ".jl")
    else
        string(join(splitted[1:end-1], "/"), "/test_", splitted[end], ".jl")
    end
end

function filename_to_key(filename)
    splitted = split(filename, "/")
    if length(splitted) === 1
        return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
    else
        path, name = splitted[1:end-1], splitted[end]
        return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
    end
end

using Aqua

if isempty(testrunner.enabled_tests)
    println("Running all tests...")
    # `project_toml_formatting` is broken on CI, revise at some point
    # TODO uncomment Aqua.test_all
    # Aqua.test_all(RxInfer; ambiguities = false, project_toml_formatting = false)
    # doctest(RxInfer)
else
    println("Running specific tests:")
    foreach(testrunner.enabled_tests) do test
        println(" - ", test)
    end
end

@testset ExtendedTestSet "RxInfer" begin
    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) ==
              "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) ==
              "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    addtests(testrunner, "test_helpers.jl")

    addtests(testrunner, "score/test_bfe.jl")

    addtests(testrunner, "constraints/test_form_constraints.jl")
    addtests(testrunner, "constraints/test_factorisation_constraints.jl")
    addtests(testrunner, "constraints/form/test_form_point_mass.jl")

    addtests(testrunner, "test_model.jl")
    addtests(testrunner, "test_inference.jl")

    addtests(testrunner, "models/aliases/test_aliases_binary.jl")
    addtests(testrunner, "models/aliases/test_aliases_normal.jl")

    addtests(testrunner, "models/autoregressive/test_ar.jl")
    addtests(testrunner, "models/autoregressive/test_lar.jl")

    addtests(testrunner, "models/mixtures/test_gmm_univariate.jl")
    addtests(testrunner, "models/mixtures/test_gmm_multivariate.jl")

    addtests(testrunner, "models/statespace/test_ulgssm.jl")
    addtests(testrunner, "models/statespace/test_mlgssm.jl")
    addtests(testrunner, "models/statespace/test_hmm.jl")

    addtests(testrunner, "models/iid/test_mv_iid_precision.jl")
    addtests(testrunner, "models/iid/test_mv_iid_precision_known_mean.jl")
    addtests(testrunner, "models/iid/test_mv_iid_covariance.jl")
    addtests(testrunner, "models/iid/test_mv_iid_covariance_known_mean.jl")

    # addtests(testrunner, "models/test_lgssm.jl")
    # addtests(testrunner, "models/test_hgf.jl")
    # addtests(testrunner, "models/test_ar.jl")
    # addtests(testrunner, "models/test_gmm.jl")
    # addtests(testrunner, "models/test_hmm.jl")
    # addtests(testrunner, "models/test_linreg.jl")
    # addtests(testrunner, "models/test_mv_iid.jl")
    # addtests(testrunner, "models/test_probit.jl")
    # addtests(testrunner, "models/test_aliases.jl")

    run(testrunner)
end
