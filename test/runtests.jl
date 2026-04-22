using Aqua, TestItemRunner, RxInfer

ENV["LOG_USING_RXINFER"] = "false"

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

if get(ENV, "RUN_AQUA", "true") == "true"
    Aqua.test_all(
        RxInfer;
        ambiguities = false,
        piracies = false,
        deps_compat = (; check_extras = false, check_weakdeps = true),
    )
end

if isempty(ARGS)
    @run_package_tests(verbose = true)
else
    filtered_files = map(arg -> join(split(arg, ":"), "/"))
    filter_fn = ti -> any(occursin(ti.filename), filtered_files)
    @run_package_tests(filter = filter_fn, verbose = true)
end
