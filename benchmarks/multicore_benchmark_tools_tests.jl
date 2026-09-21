using Test
include("multicore_benchmark_tools.jl")

@testset "Multicore BenchmarkTools harness" begin
    original_blas = BLAS.get_num_threads()
    withenv(
        "RXINFER_BENCHMARK_BLAS_THREADS" => "1",
        "RXINFER_BENCHMARK_RUNNER_BLAS_THREADS" => nothing,
        "RXINFER_BENCHMARK_OUTPUT" => nothing,
    ) do
        calls = Dict(0 => 0, 1 => 0)
        checks = Dict(0 => 0, 1 => 0)
        function run_inference(w)
            calls[w] += 1
            return w
        end
        function check_result(w, result)
            @test w == result
            @test BLAS.get_num_threads() == 1
            checks[w] += 1
        end
        trials = benchmark_multicore(
            run_inference,
            check_result;
            samples = 2,
            rounds = 2,
            workers = [0, 1],
        )
        for w in [0, 1]
            # One preflight; each round: one automatic warmup, two measured
            # evaluations and one untimed correctness check.
            @test calls[w] == 9
            @test checks[w] == 3
            for r in 1:2
                trial = trials[string(w)][string(r)]
                @test trial isa BenchmarkTools.Trial
                @test length(trial) == 2
                @test trial.params.evals == 1
                @test trial.params.gctrial
                @test trial.params.gcsample
            end
        end
        @test BLAS.get_num_threads() == original_blas
        @test_throws ErrorException benchmark_multicore(
            _ -> error("inference failed"),
            (_, _) -> nothing;
            samples = 1,
            rounds = 1,
            workers = [0],
        )
        @test BLAS.get_num_threads() == original_blas
        @test_throws ArgumentError benchmark_multicore(
            identity, (_, _) -> nothing; samples = 0
        )
        @test_throws ArgumentError benchmark_multicore(
            identity, (_, _) -> nothing; rounds = 0
        )
    end
    withenv(
        "RXINFER_BENCHMARK_BLAS_THREADS" => "2",
        "RXINFER_BENCHMARK_RUNNER_BLAS_THREADS" => "1",
        "RXINFER_BENCHMARK_OUTPUT" => nothing,
    ) do
        observed = Pair{Int, Int}[]
        run_inference(w) = push!(observed, w => BLAS.get_num_threads())
        trials = benchmark_multicore(
            run_inference,
            (_, _) -> nothing;
            samples = 1,
            rounds = 2,
            workers = [0, 1],
        )
        @test length(observed) == 14
        @test all(p -> p.second == (iszero(p.first) ? 2 : 1), observed)
        @test "standard_blas=2" in trials.tags
        @test "runner_blas=1" in trials.tags
        @test BLAS.get_num_threads() == original_blas
        @test_throws ErrorException benchmark_multicore(
            w -> w == 1 ? error("wave inference failed") : nothing,
            (_, _) -> nothing;
            samples = 1,
            rounds = 1,
            workers = [0, 1],
        )
        @test BLAS.get_num_threads() == original_blas
    end
end
