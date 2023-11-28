using Aqua, CpuId, ReTestItems, RxInfer

Aqua.test_all(RxInfer; ambiguities = false, piracies = false, deps_compat = (; check_extras = false, check_weakdeps = true))

nthreads = max(cputhreads(), 1)
ncores = max(cpucores(), 1)

runtests(RxInfer; nworkers = ncores, nworker_threads = Int(nthreads / ncores), memory_threshold = 1.0)
