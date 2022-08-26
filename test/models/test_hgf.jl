module RxInferModelsHGFTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
@model [default_factorisation = MeanField()] function hgf_online_model(real_k, real_w, z_variance, y_variance)

    # Priors from previous time step for `z`
    zt_min_mean = datavar(Float64)
    zt_min_var  = datavar(Float64)

    # Priors from previous time step for `x`
    xt_min_mean = datavar(Float64)
    xt_min_var  = datavar(Float64)

    zt_min ~ NormalMeanVariance(zt_min_mean, zt_min_var)
    xt_min ~ NormalMeanVariance(xt_min_mean, xt_min_var)

    meta = GCVMetadata(GaussHermiteCubature(9))

    # Higher layer is modelled as a random walk 
    zt ~ NormalMeanVariance(zt_min, z_variance) where {q = q(zt, zt_min)q(z_variance), meta = meta}

    # Lower layer is modelled with `GCV` node
    gcv_node, xt ~ GCV(xt_min, zt, real_k, real_w) where {q = q(xt, xt_min)q(zt)q(κ)q(ω)}

    # Noisy observations 
    y = datavar(Float64)
    y ~ NormalMeanVariance(xt, y_variance)

    return zt, xt, y, gcv_node, xt_min_mean, xt_min_var, zt_min_mean, zt_min_var
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function hgf_online_inference(data, vmp_iters, real_k, real_w, z_variance, y_variance)
    n = length(data)

    # We don't want to save all marginals from all VMP iterations
    # but only last one after all VMP iterations per time step
    # Rocket.jl exports PendingScheduler() object that postpones 
    # any update unless manual `resolve!()` has been called
    ms_scheduler = PendingScheduler()

    mz = keep(Marginal)
    mx = keep(Marginal)
    fe = ScoreActor(Float64)

    model, (zt, xt, y, gcv_node, xt_min_mean, xt_min_var, zt_min_mean, zt_min_var) =
        hgf_online_model(real_k, real_w, z_variance, y_variance)

    # Initial priors
    current_zt_mean, current_zt_var = 0.0, 10.0
    current_xt_mean, current_xt_var = 0.0, 10.0

    s_mz = subscribe!(getmarginal(zt) |> schedule_on(ms_scheduler), mz)
    s_mx = subscribe!(getmarginal(xt) |> schedule_on(ms_scheduler), mx)
    s_fe = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    # Initial marginals to start VMP procedire
    setmarginal!(gcv_node, :y_x, MvNormalMeanCovariance([0.0, 0.0], [5.0, 5.0]))
    setmarginal!(gcv_node, :z, NormalMeanVariance(0.0, 5.0))

    # For each observations we perofrm `vmp_iters` VMP iterations
    for i in 1:n
        for _ in 1:vmp_iters
            update!(y, data[i])
            update!(zt_min_mean, current_zt_mean)
            update!(zt_min_var, current_zt_var)
            update!(xt_min_mean, current_xt_mean)
            update!(xt_min_var, current_xt_var)
        end

        # After all VMP iterations we release! `PendingScheduler`
        # as well as release! `ScoreActor` to indicate new time step
        release!(ms_scheduler)
        release!(fe)

        current_zt_mean, current_zt_var = mean_var(last(mz))::Tuple{Float64, Float64}
        current_xt_mean, current_xt_var = mean_var(last(mx))::Tuple{Float64, Float64}
    end

    # It is important to unsubscribe at the end of the inference procedure
    unsubscribe!((s_mz, s_mx, s_fe))

    return map(getvalues, (mz, mx, fe))
end
## -------------------------------------------- ##

@testset "Hierarchical Gaussian Filter" begin
    @testset "Online inference" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        function generate_data(rng, k, w, zv, yv)
            z_prev = 0.0
            x_prev = 0.0

            z = Vector{Float64}(undef, n)
            v = Vector{Float64}(undef, n)
            x = Vector{Float64}(undef, n)
            y = Vector{Float64}(undef, n)

            for i in 1:n
                z[i] = rand(rng, Normal(z_prev, sqrt(zv)))
                v[i] = exp(k * z[i] + w)
                x[i] = rand(rng, Normal(x_prev, sqrt(v[i])))
                y[i] = rand(rng, Normal(x[i], sqrt(yv)))

                z_prev = z[i]
                x_prev = x[i]
            end

            return z, x, y
        end
        ## -------------------------------------------- ##
        rng = StableRNG(42)
        # Parameters of HGF process
        real_k = 1.0
        real_w = 0.0
        z_variance = abs2(0.2)
        y_variance = abs2(0.1)
        # Number of observations
        n = 2000
        z, x, y = generate_data(rng, real_k, real_w, z_variance, y_variance)
        ## -------------------------------------------- ##
        ## Inference execution
        vmp_iters = 10
        mz, mx, fe = hgf_online_inference(y, vmp_iters, real_k, real_w, z_variance, y_variance)
        ## -------------------------------------------- ##
        ## Test inference results
        @test length(mz) === n
        @test length(mx) === n
        @test length(fe) === vmp_iters
        @test abs(last(fe) - 2027.8628798126442) < 0.01
        @test all(filter(e -> abs(e) > 0.1, diff(fe)) .< 0)
        # Check if all estimates are within 6std interval
        @test all((mean.(mz) .- 6 .* std.(mz)) .< z .< (mean.(mz) .+ 6 .* std.(mz)))
        @test all((mean.(mx) .- 6 .* std.(mx)) .< x .< (mean.(mx) .+ 6 .* std.(mx)))
        # Check if more than 95% of estimates are within 3std interval
        @test (sum((mean.(mz) .- 3 .* std.(mz)) .< z .< (mean.(mz) .+ 3 .* std.(mz))) / n) > 0.95
        @test (sum((mean.(mx) .- 3 .* std.(mx)) .< x .< (mean.(mx) .+ 3 .* std.(mx))) / n) > 0.95
        @test all(var.(mx) .> 0.0)
        @test all(var.(mz) .> 0.0)
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "hgf_online_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "hgf_online_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        pz = plot(title = "Hidden States Z")
        px = plot(title = "Hidden States X")

        plot!(pz, 1:n, z, label = "z_i", color = :orange)
        plot!(pz, 1:n, mean.(mz), ribbon = std.(mz), label = "estimated z_i", color = :teal)

        plot!(px, 1:n, x, label = "x_i", color = :green)
        plot!(px, 1:n, mean.(mx), ribbon = std.(mx), label = "estimated x_i", color = :violet)

        pf = plot(fe, label = "Bethe Free Energy")

        p = plot(pz, px, pf, layout = @layout([a; b; c]))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark =
                @benchmark hgf_online_inference($y, $vmp_iters, $real_k, $real_w, $z_variance, $y_variance) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
