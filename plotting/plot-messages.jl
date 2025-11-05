### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 197500c2-25a1-427e-8c36-d5147dc4cd5a
begin
	using Pkg
	Pkg.activate(".")
	using PlutoUI
	using JLD2
	using Distributions
	#using Plots
	using BayesBase
	using ExponentialFamily
	using StatsPlots
	using GLMakie
end

# ╔═╡ d68dd5d4-b942-11f0-0d9b-d3672e202996
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(80px, 5%);
    	padding-right: max(80px, 5%);
	}
</style>
"""

# ╔═╡ 7be8295b-8dcb-4de1-a752-cc7ba513f8ce

function safe_range(d; npoints=300)
    # Get distribution support
    s = support(d)
    lo_supp, hi_supp = extrema(s)

    # Compute quantile-based or moment-based range
    try
        lo = quantile(d, 0.001)
        hi = quantile(d, 0.999)
        if !isfinite(lo) || !isfinite(hi) || lo == hi
            throw(DomainError())
        end
    catch
        try
            m, s = mean(d), std(d)
            lo, hi = m - 4s, m + 4s
        catch
            lo, hi = -10.0, 10.0
        end
    end

    # Clip to distribution support
    lo = max(lo, lo_supp)
    hi = min(hi, hi_supp)

    # Handle degenerate supports (e.g. PointMass)
    if !isfinite(lo) || !isfinite(hi) || lo == hi
        lo, hi = lo_supp, hi_supp
        if !isfinite(lo) || !isfinite(hi)
            lo, hi = -10.0, 10.0
        end
    end

    return range(lo, hi, length=npoints)
end

# ╔═╡ 8f9845fa-1bd7-4a5f-a428-c7d28e242d7d
probe_dir = "results"

# ╔═╡ 7a6decc6-0579-4ce9-af9c-275bdbcf1572
begin
probe_files = filter(f -> endswith(f, ".jld2"), readdir(probe_dir; join=true))
@bind selected_probe_file Select(
    probe_files,
    default = isempty(probe_files) ? nothing : first(probe_files),
)
end

# ╔═╡ d3fdf5ed-295a-47ec-9926-c674f957888e
probe = isnothing(selected_probe_file) ? nothing : load(selected_probe_file, "probe")

# ╔═╡ fa3bcce1-afb7-4e8c-a059-992378c12d34
@bind iter_idx PlutoUI.Slider(sort(collect(keys(probe.data))), show_value=true)

# ╔═╡ 470d8d33-f3d6-4775-bcef-65bb83a63937
@bind rec_idx  PlutoUI.Slider(1:length(probe.data[iter_idx]), show_value=true)

# ╔═╡ 15f71fc7-428a-4697-832b-3c4bd6806dad
rec = probe.data[iter_idx][rec_idx]

# ╔═╡ 2ffc742e-cfe8-463a-bf41-0578e40e2f7a
rec

# ╔═╡ 48f762ae-f0f3-42b1-a41f-54d33d774335
function plot_inputs_and_result(rec; iter_idx=1, rec_idx=1)
    fig = Figure(size=(650, 420))
    ax = Axis(
        fig[1, 1],
        title = "$(rec.node)[$(rec.interface)] (iter $iter_idx, step $rec_idx)",
        xlabel = "x",
        ylabel = "density"
    )

    # Function that handles any single distribution
    function plot_distribution!(ax, d, label; color=:gray)
        if d isa PointMass
            xval = mean(d)
            lines!(ax, [xval, xval], [0, 1],
                   linewidth=2, linestyle=:dash,
                   color=color, label=label)
        elseif hasmethod(pdf, (typeof(d), Float64))
            xs = safe_range(d)
            ys = try
                pdf.(d, xs)
            catch
                fill(NaN, length(xs))
            end
            band!(ax, xs, zeros(length(xs)), ys;
                  color=(color, 0.25)) # translucent fill
            lines!(ax, xs, ys; color=color, linewidth=2, label=label)
        else
            @warn "Cannot plot distribution of type $(typeof(d))"
        end
    end

    # Plot all input distributions
    palette = Makie.wong_colors()
    for (i, d) in enumerate(rec.inputs)
        plot_distribution!(ax, d, "Input $i"; color=palette[mod1(i, length(palette))])
    end

    # Plot the result distribution
    plot_distribution!(ax, rec.result, "Result"; color=:dodgerblue)

    axislegend(ax, position=:rt)
    return fig
end

# ╔═╡ 98060fa1-58dc-4127-9995-377fe07ab64f
rec.inputs

# ╔═╡ 80b68212-3f46-43a8-99c5-51d1b74198ba
plot_inputs_and_result(rec, iter_idx=iter_idx, rec_idx=rec_idx)

# ╔═╡ 347e47b7-d7be-4c10-98af-b191c67aa787
begin
	records = [(iter, rec) for (iter, vec) in probe.data for rec in vec]
	n_records = length(records)
	println("Loaded $n_records records across $(length(keys(probe.data))) iterations")
end

# ╔═╡ Cell order:
# ╠═d68dd5d4-b942-11f0-0d9b-d3672e202996
# ╠═197500c2-25a1-427e-8c36-d5147dc4cd5a
# ╠═7be8295b-8dcb-4de1-a752-cc7ba513f8ce
# ╠═8f9845fa-1bd7-4a5f-a428-c7d28e242d7d
# ╠═7a6decc6-0579-4ce9-af9c-275bdbcf1572
# ╠═d3fdf5ed-295a-47ec-9926-c674f957888e
# ╠═fa3bcce1-afb7-4e8c-a059-992378c12d34
# ╠═470d8d33-f3d6-4775-bcef-65bb83a63937
# ╠═15f71fc7-428a-4697-832b-3c4bd6806dad
# ╠═2ffc742e-cfe8-463a-bf41-0578e40e2f7a
# ╟─48f762ae-f0f3-42b1-a41f-54d33d774335
# ╠═98060fa1-58dc-4127-9995-377fe07ab64f
# ╠═80b68212-3f46-43a8-99c5-51d1b74198ba
# ╠═347e47b7-d7be-4c10-98af-b191c67aa787
