using Distributed

addprocs(Sys.CPU_THREADS)

const examples = RemoteChannel(() -> Channel(32));
const results  = RemoteChannel(() -> Channel(32));

@everywhere using Weave

@everywhere function process_example(examples, results) 
    pid = myid()
    while true
        try 
            example = take!(examples)
            path = example[:path]
        
            println("Started job: `$(path)` on worker `$(pid)`")

            ipath = joinpath(@__DIR__, "..", "examples", path)
            opath = joinpath(@__DIR__, "..", "docs", "src", "examples")

            put!(results, (pid = pid, error = nothing, path = path, weaved = weave(ipath, out_path = opath, doctype = "github"), example = example))
        catch e 
            @error e
            put!(results, (pid = pid, error = e, path = path, weaved = nothing, example = example))
        end
    end
end

global jobs = include(joinpath(@__DIR__, "..", "examples", ".meta.jl"))
global n    = length(jobs)

function make_jobs(jobs)
    for example in jobs 
        put!(examples, example)
    end
end

@async make_jobs(jobs)

for p in workers() # start tasks on the workers to process requests in parallel
    remote_do(process_example, p, examples, results)
end

while n > 0 # print out results
    try 
        result = take!(results)
        pid    = result[:pid]
        error  = result[:error]
        path   = result[:path]

        if isnothing(error)
            println("Finished `$(path)` on worker `$(pid)`.")
        else
            println("Failed to process `$(path) on worker `$(pid)`: $(error).")
        end
    catch e
        @error e
    end
    global n = n - 1
end

println("Finished.")