module RxInferModelsHGFTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
# We create a single-time step of corresponding state-space process to
# perform online learning (filtering)

@model function hgf(real_k, real_w, z_variance, y_variance)

    # Priors from previous time step for `z`
    zt_min_mean = datavar(Float64)
    zt_min_var  = datavar(Float64)

    # Priors from previous time step for `x`
    xt_min_mean = datavar(Float64)
    xt_min_var  = datavar(Float64)

    zt_min ~ NormalMeanVariance(zt_min_mean, zt_min_var)
    xt_min ~ NormalMeanVariance(xt_min_mean, xt_min_var)

    # Higher layer is modelled as a random walk 
    zt ~ NormalMeanVariance(zt_min, z_variance)

    # Lower layer is modelled with `GCV` node
    gcvnode, xt ~ GCV(xt_min, zt, real_k, real_w)

    # Noisy observations 
    y = datavar(Float64)
    y ~ NormalMeanVariance(xt, y_variance)

    return gcvnode
end

@constraints function hgfconstraints()
    q(xt, zt, xt_min) = q(xt, xt_min)q(zt)
end

@meta function hgfmeta()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV(xt_min, xt, zt) -> GCVMetadata(GaussHermiteCubature(31))
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function hgf_online_inference(data, vmp_iters, real_k, real_w, z_variance, y_variance)
    autoupdates = @autoupdates begin
        zt_min_mean, zt_min_var = mean_var(q(zt))
        xt_min_mean, xt_min_var = mean_var(q(xt))
    end

    result = rxinference(
        model         = hgf(real_k, real_w, z_variance, y_variance),
        constraints   = hgfconstraints(),
        meta          = hgfmeta(),
        data          = (y = data,),
        autoupdates   = autoupdates,
        keephistory   = length(data),
        historyvars   = (
        xt = KeepLast(),
        zt = KeepLast()
    ),
        initmarginals = (
        zt = NormalMeanVariance(0.0, 5.0),
        xt = NormalMeanVariance(0.0, 5.0)
    ),
        iterations    = vmp_iters,
        free_energy   = true,
        autostart     = true,
        callbacks     = (
        after_model_creation = (model, returnval) -> begin
            gcvnode = returnval
            setmarginal!(gcvnode, :y_x, MvNormalMeanCovariance([0.0, 0.0], [5.0, 5.0]))
        end,
    )
    )

    mz = result.history[:zt]
    mx = result.history[:xt]

    # TODO Free energy history
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
