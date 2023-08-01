using Distributed

const ExamplesFolder = joinpath(@__DIR__, "..", "examples")

import Pkg;
Pkg.activate(ExamplesFolder);
Pkg.instantiate();

# Precompile packages on the main worker
@info "Precompile packages on the main worker"
using RxInfer, Plots, StatsPlots, BenchmarkTools, ProgressMeter, Optim

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

    configuration = include(joinpath(@__DIR__, "..", "examples", ".meta.jl"))

    if !haskey(configuration, :categories)
        error("The `.meta.jl` should return a configuration object with the `categories` field.")
    end

    if !haskey(configuration, :examples)
        error("The `.meta.jl` should return a configuration object with the `examples` field.")
    end

    categories = configuration[:categories]
    examples = configuration[:examples]

    if !isnothing(examplesrunner.specific_example)
        @info "Running specific example matching the following pattern: $(examplesrunner.specific_example)."
        examples = filter(examples) do example
            return occursin(lowercase(examplesrunner.specific_example), lowercase(example[:filename])) ||
                   occursin(lowercase(examplesrunner.specific_example), lowercase(example[:title]))
        end
    end

    if isempty(examples)
        error("The list of examples is empty.")
    end

    foreach(examples) do example
        @info "Adding $(example[:filename]) to the jobs list"
        put!(examplesrunner.jobschannel, example)
    end

    @info "Preparing `examples` environment."

    efolder = joinpath(@__DIR__, "..", "examples")
    dfolder = joinpath(@__DIR__, "..", "docs", "src", "examples")
    afolder = joinpath(@__DIR__, "..", "docs", "src", "assets", "examples")

    # Make folder for the documentation
    mkpath(dfolder)

    # `categories` field is a `label => info` pairs, where `label` is assumed to be a folder name
    for (label, _) in pairs(categories)
        # Make folder for each category
        mkpath(joinpath(dfolder, string(label)))
    end

    # Make path for pictures
    mkpath(joinpath(efolder, "pics"))
    mkpath(joinpath(dfolder, "pics"))
    mkpath(joinpath(afolder, "pics"))

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
                        filename = example[:filename]

                        # Check if the example has category with it
                        if !haskey(example, :category)
                            error("Missing category for the example $(filename).")
                        end

                        category = string(example[:category])

                        @info "Started job: `$(filename)` on worker `$(pid)`"

                        ipath = joinpath(@__DIR__, "..", "examples", category, filename)
                        opath = joinpath(@__DIR__, "..", "docs", "src", "examples", category)
                        fpath = joinpath("..", "..", "assets", "examples") # relative to `opath`

                        ENV["GKSwstype"] = "nul" # Fix for plots

                        weaved = weave(ipath, out_path = opath, doctype = "github", fig_path = fpath)

                        put!(resultschannel, (pid = pid, filename = filename, weaved = weaved, example = example))
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
            pid = result[:pid]
            filename = result[:filename]
            @info "Finished `$(filename)` on worker `$(pid)`."
            push!(results, result)
        end

    else
        @error "No example have been generated"
        exit(-1)
    end

    close(examplesrunner.exschannel)
    close(examplesrunner.jobschannel)
    close(examplesrunner.resultschannel)

    # Fix paths from the `pics/` folder located in the examples
    fixpics = (
        "![](pics/" => "![](../assets/examples/pics/", 
        "![](./pics/" => "![](../assets/examples/pics/",
        "![](../pics/" => "![](../assets/examples/pics/"
    )

    if isnothing(examplesrunner.specific_example)

        # If not failed we generate overview report and fix fig links
        io_main_overview = IOBuffer() # main overview
        io_category_overviews = map(category -> (io = IOBuffer(), category = category), categories) # sub-category overviews

        @info "Generating overviews"

        write(io_main_overview, "# [Examples overview](@id examples-overview)\n\n")
        write(io_main_overview, "This section contains a set of examples for Bayesian Inference with `RxInfer` package in various probabilistic models.\n\n")
        write(io_main_overview, "!!! note\n")
        write(
            io_main_overview,
            "\tAll examples have been pre-generated automatically from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n"
        )

        foreach(pairs(io_category_overviews)) do (label, overview)
            if !isequal(label, :hidden_examples)
                # Add a small description to the main overview file
                write(io_main_overview, "- [$(overview.category.title)](@ref examples-$(label)-overview): $(overview.category.description)\n")

                # Write sub descriptions in each distinct sub category
                write(overview.io, "# [$(overview.category.title)](@id examples-$(label)-overview)\n\n")
                write(overview.io, "This section contains a set of examples for Bayesian Inference with `RxInfer` package in various probabilistic models.\n\n")
                write(overview.io, "!!! note\n")
                write(
                    overview.io,
                    "\tAll examples have been pre-generated automatically from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n"
                )
                write(overview.io, "$(overview.category.description)\n\n")
            end
        end

        foreach(examples) do example
            mdname = replace(example[:filename], ".ipynb" => ".md")
            mdpath = joinpath(@__DIR__, "..", "docs", "src", "examples", string(example[:category]), mdname)
            mdtext = read(mdpath, String)

            # Check if example failed with an error
            # TODO: we might have better heurstic here? But I couldn't find a way to tell `Weave.jl` if an error has occured
            # TODO: try to improve this later
            erroridx = findnext("```\nError:", mdtext, 1)
            if !isnothing(erroridx)
                @error "`Error` block found in the `$(mdpath)` example. Check the logs for more details."
                # We print a part of the file, which (hopefully) should be enough to identify the issue
                # For more logs check the actual output
                errwindow = 500 # We need to iterate over the text with `prevind` and `nextind`, because strings are UTF8
                errstart = reduce((idx, _) -> max(firstindex(mdtext), prevind(mdtext, idx)), 1:errwindow; init = first(erroridx))
                errend = reduce((idx, _) -> min(lastindex(mdtext), nextind(mdtext, idx)), 1:errwindow; init = last(erroridx))
                @error "Part of the error message:\n\n$(mdtext[errstart:errend])\n"
                error(-1)
            end

            # We simply remove pre-generated `.md` file if it has been marked as hidden
            if isequal(example[:category], :hidden_examples)
                @info "Skipping example $(example[:title]) as it has been marked as hidden."
                rm(mdpath, force = true)
                return nothing
            end

            title       = example[:title]
            description = example[:description]
            id          = string("examples-", lowercase(join(split(example[:title]), "-")))

            if isnothing(findnext("# $(title)", mdtext, 1))
                error("Could not find cell `# $(title)` in the `$(mdpath)`")
            end

            open(mdpath, "w") do f
                # In every examples we replace title with its `@id` equivalent, such that 
                # `# Super cool title` becomes `[# Super cool title](@id examples-super-cool-title)`
                fixtext = replace(mdtext, "# $(title)" => "# [$(title)](@id $(id))", fixpics...)
                output  = string("This example has been auto-generated from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n", fixtext)
                write(f, output)
            end

            write(io_category_overviews[example.category].io, "- [$(title)](@ref $id): $description\n")

            return nothing
        end

        # Copy the `pics` folder from the examples to the assets
        if isdir(joinpath(efolder, "pics"))
            @info "Copying the `pics` folder from the examples."
            cp(joinpath(efolder, "pics"), joinpath(afolder, "pics"); force = true)
        end

        # Write main overview
        open(joinpath(@__DIR__, "..", "docs", "src", "examples", "overview.md"), "w") do f
            write(f, String(take!(io_main_overview)))
        end

        # Write sub-categories overviews
        foreach(pairs(io_category_overviews)) do (label, overview)
            if !isequal(label, :hidden_examples)
                open(joinpath(@__DIR__, "..", "docs", "src", "examples", string(label), "overview.md"), "w") do f
                    write(f, String(take!(overview.io)))
                end
            end
        end

    else
        @info "Skip overview generation for a specific example. Possible errors in the generated document are supressed. Check the generated document manually."
    end

    # `Weave` executes notebooks in the `dst` folder so we need to copy there our environment (and remove it)
    rm(joinpath(dfolder, "Manifest.toml"), force = true)
    rm(joinpath(dfolder, "Project.toml"), force = true)
    rm(joinpath(dfolder, "data"), force = true, recursive = true)

    @info "Finished."
end

const runner = ExamplesRunner(ARGS)

run(runner)
