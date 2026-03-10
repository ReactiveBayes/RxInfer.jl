using Aqua, Hwloc, ReTestItems, RxInfer

const IS_USE_DEV = get(ENV, "USE_DEV", "false") == "true"
const IS_BENCHMARK = get(ENV, "BENCHMARK", "false") == "true"

ENV["LOG_USING_RXINFER"] = "false"

import Pkg

if IS_USE_DEV
    Pkg.rm("ReactiveMP")
    Pkg.rm("GraphPPL")
    Pkg.rm("Rocket")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(Pkg.devdir(), "ReactiveMP.jl")))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(Pkg.devdir(), "GraphPPL.jl")))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(Pkg.devdir(), "Rocket.jl")))
    Pkg.resolve()
    Pkg.update()
end

if get(ENV, "RUN_AQUA", "true") == "true"
    Aqua.test_all(RxInfer; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))
end

nthreads, ncores = Hwloc.num_virtual_cores(), Hwloc.num_physical_cores()
nthreads, ncores = max(nthreads, 1), max(ncores, 1)
nworker_threads = Int(nthreads / ncores)
memory_threshold = 1.0

# We use only `1` runner in case if benchmarks are enabled to improve the 
# quality of the benchmarking procedure
if IS_BENCHMARK
    nthreads = 1
    ncores = 1
end

pkg_root = dirname(pathof(RxInfer)) |> dirname
test_root = joinpath(pkg_root, "test")

if isempty(ARGS)
    runtests(RxInfer; nworkers = ncores, nworker_threads = nworker_threads, memory_threshold = memory_threshold)
else
    for arg in ARGS
        # Translate colon syntax (e.g., rules:normal_mean_variance → rules/normal_mean_variance)
        candidate = join(split(arg, ":"), "/")

        # Build possible test paths relative to the package test directory
        paths = [joinpath(test_root, candidate), joinpath(test_root, candidate * ".jl")]

        path = findfirst(ispath, paths)

        if path !== nothing
            selected_path = paths[path]
            @info "Running selective tests from $selected_path"
            runtests(selected_path; nworkers = ncores, nworker_threads = nworker_threads, memory_threshold = memory_threshold)
        else
            @warn "Test target not found: $arg"
        end
    end
end
