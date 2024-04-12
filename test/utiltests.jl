import Dates: format, now
using MacroTools

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
        plotfn = () -> begin
            $code
        end
        savefig(plotfn(), plot_output)
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
