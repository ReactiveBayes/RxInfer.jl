using Distributed

addprocs(Sys.CPU_THREADS, exeflags="--project=$(joinpath(@__DIR__, "..", "examples"))")

const examples = RemoteChannel(() -> Channel(128)); # The number should be larger than number of examples
const results  = RemoteChannel(() -> Channel(128));

@everywhere using Weave
@everywhere using RxInfer

# Each worker listens to the `examples` channel and takes available job as soon as possibles
@everywhere function process_example(examples, results) 
    pid = myid()
    while true
        try 
            example = take!(examples)
            path = example[:path]
        
            @info "Started job: `$(path)` on worker `$(pid)`"

            ipath  = joinpath(@__DIR__, "..", "examples", path)
            opath  = joinpath(@__DIR__, "..", "docs", "src", "examples")
            fpath  = joinpath("..", "assets", "examples") # relative to `opath`
            weaved = weave(ipath, out_path = opath, doctype = "multimarkdown", fig_path = fpath)

            put!(results, (pid = pid, error = nothing, path = path, weaved = weaved, example = example))
        catch e 
            @error e
            put!(results, (pid = pid, error = e, path = path, weaved = nothing, example = example))
        end
    end
end

function main()
    jobs      = include(joinpath(@__DIR__, "..", "examples", ".meta.jl"))
    remaining = length(jobs)

    foreach((job) -> @info("Adding $(job[:title]) at $(job[:path])."), jobs) 
    foreach((job) -> put!(examples, job), jobs)

    for p in workers() # Start tasks on the workers to process requests in parallel
        remote_do(process_example, p, examples, results)
    end

    isfailed = false

    # We wait until we process all `remaining` examples
    # We record any failure in `isfailed` variable 
    while remaining > 0 
        try 
            result = take!(results)
            pid    = result[:pid]
            error  = result[:error]
            path   = result[:path]

            if isnothing(error)
                @info "Finished `$(path)` on worker `$(pid)`."
            else
                @info "Failed to process `$(path) on worker `$(pid)`: $(error)."
            end
        catch e
            isfailed = true
            @error e
        end
        remaining -= 1
    end

    if isfailed
        @error "Examples compilation failed."
        error(-1)
    end

    # If not failed we generate overview report and fix fig links
    io_overview = IOBuffer()

    @info "Generating overview"

    write(io_overview, "# [Examples overview](@id examples-overview)\n\n")
    write(io_overview, "This section contains a set of examples for Bayesian Inference with `ReactiveMP` package in various probabilistic models.\n\n")
    write(io_overview, "!!! note\n")
    write(io_overview, "\tAll examples have been pre-generated automatically from the [`examples/`](https://github.com/biaslab/RxInfer.jl/tree/main/examples) folder at GitHub repository.\n\n")

    foreach(jobs) do job
        mdname      = replace(job[:path], ".ipynb" => ".md")
        mdpath      = joinpath(@__DIR__, "..", "docs", "src", "examples", mdname)

        if job[:hidden]
            @info "Skipping example $(job[:title]) as it has been marked as hidden"
            rm(mdpath, force = true)
            return nothing
        end

        mdtext      = read(mdpath, String)
        title       = job[:title]
        description = job[:description]
        id          = string("examples-", lowercase(join(split(job[:title]), "-")))

        if isnothing(findnext("# $(title)", mdtext, 1))
            @error "Could not find cell `# $(title)` in the `$(mdpath)`"
            error(-1)
        end

        open(mdpath, "w") do f
            # In every examples we replace title with its `@id` equivalent, such that 
            # `# Super cool title` becomes `[# Super cool title](@id examples-super-cool-title)`
            # We also fix figure links
            write(f, replace(mdtext, "# $(title)" => "# [$(title)](@id $(id))"))
        end

        write(io_overview, "- [$(title)](@ref $id): $description")

        return nothing
    end

    open(joinpath(@__DIR__, "..", "docs", "src", "examples", "overview.md"), "w") do f
        write(f, String(take!(io_overview)))
    end

    @info "Finished."
end

main()