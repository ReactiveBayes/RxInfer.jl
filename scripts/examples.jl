using Distributed

const ExamplesFolder = joinpath(@__DIR__, "..", "examples")

import Pkg;
Pkg.activate(ExamplesFolder);
Pkg.instantiate();

# Precompile packages on the main worker
@info "Precompile packages on the main worker"
using RxInfer, Plots, PyPlot, StatsPlots, BenchmarkTools, ProgressMeter, Optim

@info "Adding $(min(Sys.CPU_THREADS, 4)) workers"
addprocs(min(Sys.CPU_THREADS, 4), exeflags = "--project=$(ExamplesFolder)")

@everywhere using Weave
@everywhere using RxInfer

struct ExamplesRunner
    specific_example
    runner_tasks
    workerpool
    jobschannel
    resultschannel
    exschannel

    function ExamplesRunner(ARGS)
        specific_example = isempty(ARGS) ? nothing : first(ARGS)
        runner_tasks = []
        jobschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for jobs
        resultschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for results
        exschannel = RemoteChannel(() -> Channel(Inf), myid()) # Channel for exceptions
        return new(specific_example, runner_tasks, 2:nprocs(), jobschannel, resultschannel, exschannel)
    end
end

function Base.run(examplesrunner::ExamplesRunner)
    @info "Reading .meta.jl"

    examples = include(joinpath(@__DIR__, "..", "examples", ".meta.jl"))

    if !isnothing(examplesrunner.specific_example)
        @info "Running specific example matching the following pattern: $(examplesrunner.specific_example)"
        examples = filter(examples) do example
            return occursin(lowercase(examplesrunner.specific_example), lowercase(example[:path])) || occursin(lowercase(examplesrunner.specific_example), lowercase(example[:title]))
        end
    end

    if isempty(examples)
        @error "Examples list is empty"
        exit(-1)
    end

    foreach(examples) do example
        @info "Adding $(example[:path]) to the jobs list"
        put!(examplesrunner.jobschannel, example)
    end

    @info "Preparing `examples` environment"

    efolder = joinpath(@__DIR__, "..", "examples")
    dfolder = joinpath(@__DIR__, "..", "docs", "src", "examples")

    mkpath(dfolder)

    # `Weave` executes notebooks in the `dst` folder so we need to copy there our environment
    cp(joinpath(efolder, "Manifest.toml"), joinpath(dfolder, "Manifest.toml"), force = true)
    cp(joinpath(efolder, "Project.toml"), joinpath(dfolder, "Project.toml"), force = true)
    cp(joinpath(efolder, "data"), joinpath(dfolder, "data"), force = true)

    # We also need to fix relative RxInfer path in the moved `Project.toml`
    # This is a bit iffy, but does the job (not sure about Windows though?)
    manifest = read(joinpath(dfolder, "Manifest.toml"), String)
    manifest = replace(manifest, "path = \"..\"" => "path = \"../../..\"")

    open(joinpath(dfolder, "Manifest.toml"), "w") do f
        write(f, manifest)
    end

    foreach(examplesrunner.workerpool) do worker
        # For each worker we create a `nothing` token in the `jobschannel`
        # This token indicates that there are no other jobs left
        put!(examplesrunner.jobschannel, nothing)
        # We create a remote call for another Julia process to execute the example
        task = remotecall(worker, examplesrunner.jobschannel, examplesrunner.resultschannel, examplesrunner.exschannel) do jobschannel, resultschannel, exschannel
            pid = myid()
            finish = false
            while !finish
                # Each worker takes examples sequentially from the shared examples pool 
                example = take!(jobschannel)
                if isnothing(example)
                    finish = true
                else
                    try
                        path = example[:path]

                        @info "Started job: `$(path)` on worker `$(pid)`"

                        ipath = joinpath(@__DIR__, "..", "examples", path)
                        opath = joinpath(@__DIR__, "..", "docs", "src", "examples")
                        fpath = joinpath("..", "assets", "examples") # relative to `opath`

                        ENV["GKSwstype"] = "nul" # Fix for plots

                        weaved = weave(ipath, out_path = opath, doctype = "github", fig_path = fpath)

                        put!(resultschannel, (pid = pid, path = path, weaved = weaved, example = example))
                    catch iexception
                        put!(exschannel, iexception)
                    end
                end
            end
        end
        # We save the created task for later syncronization
        push!(examplesrunner.runner_tasks, task)
    end

    # For each remotelly called task we `fetch` its result or save an exception
    foreach(fetch, examplesrunner.runner_tasks)

    # If exception are not empty we notify the user and force-fail
    if isready(examplesrunner.exschannel)
        @error "Tests have failed with the following exceptions: "
        while isready(examplesrunner.exschannel)
            exception = take!(examplesrunner.exschannel)
            showerror(stderr, exception)
            println(stderr, "\n", "="^80)
        end
        exit(-1)
    end

    results = []

    # At last we check the output of each example
    if isready(examplesrunner.resultschannel)
        @info "Reading results"

        while isready(examplesrunner.resultschannel)
            result = take!(examplesrunner.resultschannel)
            pid    = result[:pid]
            path   = result[:path]
            @info "Finished `$(path)` on worker `$(pid)`."
            push!(results, result)
        end

    else
        @error "No example have been generated"
        exit(-1)
    end

    close(examplesrunner.exschannel)
    close(examplesrunner.jobschannel)
    close(examplesrunner.resultschannel)

    if isnothing(examplesrunner.specific_example)

        # If not failed we generate overview report and fix fig links
        io_overview = IOBuffer()

        @info "Generating overview"

        write(io_overview, "# [Examples overview](@id examples-overview)\n\n")
        write(io_overview, "This section contains a set of examples for Bayesian Inference with `RxInfer` package in various probabilistic models.\n\n")
        write(io_overview, "!!! note\n")
        write(
            io_overview,
            "\tAll examples have been pre-generated automatically from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n"
        )

        foreach(examples) do example
            mdname = replace(example[:path], ".ipynb" => ".md")
            mdpath = joinpath(@__DIR__, "..", "docs", "src", "examples", mdname)
            mdtext = read(mdpath, String)

            # Check if example failed with an error
            # TODO: we might have better heurstic here? But I couldn't find a way to tell `Weave.jl` if an error has occured
            # TODO: try to improve this later
            if !isnothing(findnext("```\nError:", mdtext, 1))
                @error "`Error` block found in the `$(mdpath)` example. Check the logs for more details."
                error(-1)
            end

            # We simply remove pre-generated `.md` file if it has been marked as hidden
            if example[:hidden]
                @info "Skipping example $(example[:title]) as it has been marked as hidden"
                rm(mdpath, force = true)
                return nothing
            end

            title       = example[:title]
            description = example[:description]
            id          = string("examples-", lowercase(join(split(example[:title]), "-")))

            if isnothing(findnext("# $(title)", mdtext, 1))
                @error "Could not find cell `# $(title)` in the `$(mdpath)`"
                error(-1)
            end

            open(mdpath, "w") do f
                # In every examples we replace title with its `@id` equivalent, such that 
                # `# Super cool title` becomes `[# Super cool title](@id examples-super-cool-title)`
                fixid  = replace(mdtext, "# $(title)" => "# [$(title)](@id $(id))")
                output = string("This example has been auto-generated from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n", fixid)
                write(f, output)
            end

            write(io_overview, "- [$(title)](@ref $id): $description\n")

            return nothing
        end

        open(joinpath(@__DIR__, "..", "docs", "src", "examples", "Overview.md"), "w") do f
            write(f, String(take!(io_overview)))
        end
    else
        @info "Skip overview generation for a specific example"
    end

    # `Weave` executes notebooks in the `dst` folder so we need to copy there our environment (and remove it)
    rm(joinpath(dfolder, "Manifest.toml"), force = true)
    rm(joinpath(dfolder, "Project.toml"), force = true)
    rm(joinpath(dfolder, "data"), force = true, recursive = true)

    @info "Finished."
end

const runner = ExamplesRunner(ARGS)

run(runner)
