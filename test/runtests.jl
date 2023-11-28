using Aqua, CpuId, ReTestItems, RxInfer

const IS_USE_DEV = get(ENV, "USE_DEV", "false") == "true"
const IS_BENCHMARK = get(ENV, "BENCHMARK", "false") == "true"

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

Aqua.test_all(RxInfer; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

# We use only `1` runner in case if benchmarks are enabled to improve the 
# quality of the benchmarking procedure
if IS_BENCHMARK
    nthreads = 1
    ncores = 1
end

runtests(RxInfer; nworkers = ncores, nworker_threads = Int(nthreads / ncores), memory_threshold = 1.0)
