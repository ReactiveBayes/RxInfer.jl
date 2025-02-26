import Dates: format, now
using MacroTools

ENV["LOG_USING_RXINFER"] = "false"

# Some tests use plotting, this part contains helper functions to simplify executing plots

"""
    @test_plots group id code

This function executes plots with `code`. Last statement in the `code` must be `return plot`.
The results are stored in a file under `_output/<group>/plot_<id>_<timestamp>_v<version>.png`
"""
macro test_plot(group, id, code)
    timestamp  = format(now(), "dd-mm-yyyy-HH-MM")
    outputfile = "plot_$(id)_$(timestamp)_v$(VERSION).png"
    dir        = @__DIR__
    ret        = quote
        base_output = joinpath($dir, "_output", $group)
        mkpath(base_output)
        plot_output = joinpath(base_output, $outputfile)
        # recent.png output can be used to rerun tests and see 
        # the recent picture update automatical in a VScode tab
        plot_output_debug = joinpath(base_output, "recent.png")
        plotfn = () -> begin
            $code
        end
        local __p = plotfn()
        savefig(__p, plot_output)
        savefig(__p, plot_output_debug)
        nothing
    end
    return esc(ret)
end

# Some tests use benchmarking, this part contains helper functions to simplify running benchmarks
# Benchmarks in tests run only if `ENV["BENCHMARK"] = "true"` and are stored in the `_output` folder

"""
    @test_benchmark group id code

This function runs test benchmark for `code` if `ENV["BENCHMARK"]` has been set to `"true"`. 
The results are stored in a file under `_output/<group>/benchmark_<id>_<timestamp>_v<version>.txt`
"""
macro test_benchmark(group, id, code)
    timestamp  = format(now(), "dd-mm-yyyy-HH-MM")
    outputfile = "benchmark_$(id)_$(timestamp)_v$(VERSION).txt"
    dir        = @__DIR__
    ret        = quote
        if get(ENV, "BENCHMARK", nothing) == "true"
            base_output = joinpath($dir, "_output", $group)
            mkpath(base_output)
            benchmark_output = joinpath(base_output, $outputfile)
            benchmark = @benchmark begin
                $code
            end seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
    end
    return esc(ret)
end

export @test_expression_generating

macro test_expression_generating(lhs, rhs)
    test_expr_gen = gensym(:text_expr_gen)
    return esc(
        quote
            $test_expr_gen = (prettify($lhs) == prettify($rhs))
            if !$test_expr_gen
                println("Expressions do not match: ")
                println("lhs: ", prettify($lhs))
                println("rhs: ", prettify($rhs))
            end
            @test (prettify($lhs) == prettify($rhs))
        end
    )
end

function generate_multinomial_data(rng = StableRNG(123); N = 3, k = 3, nsamples = 5000)
    ψ = randn(rng, k)
    p = ReactiveMP.softmax(ψ)

    X = rand(rng, Multinomial(N, p), nsamples)
    X = [X[:, i] for i in axes(X, 2)]
    return X, ψ, p
end

function logistic_stic_breaking(m)
    Km1 = length(m)

    p = Array{Float64}(undef, Km1 + 1)
    p[1] = logistic(m[1])
    for i in 2:Km1
        p[i] = logistic(m[i]) * (1 - sum(p[1:(i - 1)]))
    end
    p[end] = 1 - sum(p[1:(end - 1)])
    return p
end
