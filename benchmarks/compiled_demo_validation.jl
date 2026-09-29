# Untimed calibration/verification for the short, shareable example.
include("../../RxInferExamples.jl/compiled_runner_demo.jl")
using Test

function validate_demo()
    y = sin.(1:5000)
    expected_counts = (162, 167, 167)
    configurations = ((limit_stack_depth=100,), (runner=CompiledRunner(workers=1),),
        (runner=CompiledRunner(),))
    reference = fit_signal(y, first(configurations), 400)
    @testset "Shareable compiled-runner demo" begin
        for (options, expected) in zip(configurations, expected_counts)
            history = fit_signal(y, options, 200; keep=KeepEach())
            previous_means, previous_vars = zeros(length(y)), zeros(length(y))
            stable, first_converged = 0, nothing
            for (iteration, q) in enumerate(history)
                means, vars = mean.(q), var.(q)
                settled = all(isapprox.(means, previous_means; rtol=1e-4, atol=1e-6)) &&
                    all(isapprox.(vars, previous_vars; rtol=1e-4, atol=1e-6))
                stable = settled ? stable + 1 : 0
                if stable >= 5 && relative_residual(y, means) <= 1e-6
                    first_converged = iteration
                    break
                end
                previous_means, previous_vars = means, vars
            end
            @test first_converged === expected
            # KeepLast has a different retention path; verify the public demo
            # call, not only the history used for calibration.
            q = fit_signal(y, options, expected)
            @test relative_residual(y, mean.(q)) <= 1e-6
            for statistic in (mean, var)
                @test all(isapprox.(statistic.(q), statistic.(reference); rtol=1e-4, atol=1e-6))
            end
            @test all(>(0), var.(q))
            println("CALIBRATED ", options, " sweeps=", first_converged,
                " residual=", relative_residual(y, mean.(q)), " stable=", stable)
            flush(stdout)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && validate_demo()
