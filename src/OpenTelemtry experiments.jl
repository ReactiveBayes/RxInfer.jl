### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ c64de8a5-71fc-4a31-8fe7-953b73158eda
import Pkg; Pkg.activate(); Pkg.resolve(); Pkg.instantiate()

# ╔═╡ 28c0af84-e133-4e70-8728-4846c80d1add
using Revise

# ╔═╡ c5bf6e4a-bd50-4d45-9e13-b2f39c730e08
using RxInfer, StableRNGs, Plots

# ╔═╡ d13c9fbb-0771-4d8e-82ae-563875cad3ce
using OrderedCollections

# ╔═╡ 866fad2d-0d4d-4ff9-a355-4c0c1cbbf6bb
using Dates

# ╔═╡ 8e8141e0-2b74-4813-ac36-2dcb6e86df47
using UUIDs

# ╔═╡ 93480c78-5e9c-4398-b2e4-414249553545
using PlutoUI

# ╔═╡ 7883817d-d74d-40c8-b1a2-08904fdc0f9c
using BenchmarkTools

# ╔═╡ 19609cfa-f124-4060-976f-40102d7cc45d
using Profile, ProfileSVG

# ╔═╡ 42f96fd4-8c47-4c41-895d-aeecbb3aa19a
using Base64

# ╔═╡ 18d21cb8-e14f-4cbd-b679-7c696eac8b21
TableOfContents()

# ╔═╡ 1e8a46ca-268c-11f1-8687-1362ac8b895a
md"""
# This notebook uses RxInfer 5.0

This notebook requires you to have locally installed version of RxInfer and ReactiveMP. Before running the code in this notebook make sure that:

### 1. You have locally installed ReactiveMP and switched to the `release-6` branch

```
cd MyProjects # or whatever folder you're using for developing
git clone git@github.com:ReactiveBayes/ReactiveMP.jl.git
cd ReactiveMP
git pull
git checkout release-6
```

### 2. You have locally installed RxInfer and switched to the `release-5` branch

```
cd MyProjects # or whatever folder you're using for developing
git clone git@github.com:ReactiveBayes/RxInfer.jl.git
cd RxInfer
git pull
git checkout release-5
```

!!! note
	`release-5` and `release-6` is not a typo, ReactiveMP and RxInfer have different major versions.

### 3. You've added local version of ReactiveMP to RxInfer dependencies

```
cd MyProjects
cd RxInfer
julia --project -e 'using Pkg; Pkg.develop(path="../ReactiveMP.jl/")'
```

### 4. Make sure Revise is installed globally

```
julia -e 'using Pkg; Pkg.add("Revise")'
```

!!! note
	The notebooks also uses packages like `StableRNGs`, `Plots`. Make sure to install them globally as well.
"""

# ╔═╡ c29b75a6-fdac-45e7-a69f-583dbad8c60d
md"""
# Inference example
"""

# ╔═╡ 45f9d11c-6c53-4294-9e64-31b10bb7a7f8
@model function iid_estimation(y)
    μ  ~ Normal(mean = 0.0, precision = 0.1)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end

# ╔═╡ 2b449df0-414e-41ea-889a-cdb399bbb61c
# Specify mean-field constraint over the joint variational posterior
constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)
end

# ╔═╡ aa0c87e6-2a10-48e9-ae1e-f92618cf90b8
# Specify initial posteriors for variational iterations
initialization = @initialization begin
    q(μ) = vague(NormalMeanPrecision)
    q(τ) = vague(GammaShapeRate)
end

# ╔═╡ f63901f0-edd6-40f0-9537-9e37f2d38c4b
begin
	hidden_μ       = 3.1415
	hidden_τ       = 2.7182
	distribution   = NormalMeanPrecision(hidden_μ, hidden_τ)
	rng            = StableRNG(42)
	n_observations = 1000
	dataset        = rand(rng, distribution, n_observations)
end

# ╔═╡ 61fcb0f3-4ff0-4cbb-bb72-8b6f22457902
results = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 4,
    initialization = initialization,
	trace          = true
)

# ╔═╡ 4cb2aaca-01eb-4ab5-9223-d59514774d18
md"""
# Getting traces from RxInfer
"""

# ╔═╡ e01f8c2a-2b13-403a-bd0d-7751567383e7
trace = results.model.metadata[:trace]

# ╔═╡ ac9085bf-4463-4f71-ae5e-8f4b59e056c9
after_marginal_events = RxInfer.tracedevents(trace)

# ╔═╡ 66399916-d2d5-4e18-8b84-ef1e10074b49
md"""
# Matching spans and finding trace hierarchy

Needed for OTel
"""

# ╔═╡ 4dc178b5-d0a0-46ac-92b8-cbe45e7d0b3a
# function get_trace_id(te::TracedEvent)
# 	event = te.event
# 	if hasfield(typeof(event), :trace_id)
# 		tid = event.trace_id
# 	end
# end

# ╔═╡ b8fdcbe7-2ed7-4cd1-b215-cb0905b7efe2
@generated function get_trace_id(event::ReactiveMP.Event)
	if hasfield(event, :trace_id)
		:(event.trace_id)
	end
end

# ╔═╡ d139d769-d411-4718-ab0a-cb6f176afd9d
get_trace_id(after_marginal_events[1].event)

# ╔═╡ 2542529b-f49c-4167-a6a3-8a099ee942d0
function find_spans(trace_events::Vector{TracedEvent})
	# Found spans
	starts = OrderedDict{Base.UUID, TracedEvent}()
	stops = OrderedDict{Base.UUID, TracedEvent}()
	
	# Currently waiting to be matched
	# lonely = OrderedDict{Base.UUID, Any}()
	
	# No match possible
	not_matched = TracedEvent[]

	for te in trace_events
		tid = get_trace_id(te.event)
		if !isnothing(tid)
			if haskey(starts, tid)
				stops[tid] = te
			else
				starts[tid] = te
			end
		else
			push!(not_matched, te)
		end
	end

	push!(not_matched, setdiff(keys(starts), keys(stops))...)

	spans = [
		(starts[k], stops[k])
		for k in collect(keys(starts)) ∩ keys(stops)
	]

	return (; spans, not_matched )
end

# ╔═╡ 98d10049-164f-44b4-8c6c-6a1714aa0a3d
found = find_spans(after_marginal_events)

# ╔═╡ d12d24be-938c-4fd6-918d-955427ab8b50
md"""
## Finding hierarchy
"""

# ╔═╡ cf550cc4-5dd8-4340-875c-2032441ce765
# ╠═╡ disabled = true
#=╠═╡
function find_parent(span, all_spans)
	start = findfirst(isequal(span), all_spans)
	is_bigger(other_span) = 
		other_span[1].time_ns <= span[1].time_ns &&
		other_span[2].time_ns >= span[2].time_ns

	other_indices = (start-1):-1:1
	r = findfirst(i -> is_bigger(all_spans[i]), other_indices)
	r === nothing ? nothing : other_indices[r]
end
  ╠═╡ =#

# ╔═╡ 04407b00-3308-4e35-b144-4dac46afc252
#=╠═╡
zzz = map(found.spans) do s
	find_parent(s, found.spans)
end
  ╠═╡ =#

# ╔═╡ f479106f-d65c-479a-8ef6-3fe8dbe391b7
md"""
# OTLP JSON

Jaeger's "Upload" feature expects the OTLP JSON format.
Spec: https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding

Key idea: timestamps are nanoseconds as *strings*, 
trace/span IDs are hex strings.
"""

# ╔═╡ 252de4a1-24d7-4c84-8196-44e0fb50b37b
import JSON3

# ╔═╡ 1afa52be-6a37-4049-b0f9-e9d0d720f31f


# ╔═╡ 0747b0f9-1081-4bb9-8193-6d3d2c59cda4
dump(quote
	x = (a=1,b=2)
end)

# ╔═╡ 299a9c0d-1ef6-421d-ba97-317a913b1940
VERSION

# ╔═╡ 2c049a69-24d1-43ab-bd18-4ab9828628df


# ╔═╡ 57251562-4f84-4dd5-b0af-9000983366d9
function to_observability_string(z::Any, keyname::Symbol)

	# tstart = time()
	# result = string(z)
	# elapsed = time() - tstart

	# if elapsed > 0.003
	# 	@warn "Field took too long to render to string" elapsed typeof(z) keyname result
	# end

	
	# if length(result) > 2000
	# 	@warn "String result is very long" elapsed typeof(z) keyname result
	# end

	# result

	string(z)
end

# ╔═╡ 31455dd1-1755-46ed-995a-00c17d1653b2
to_observability_string(z::ReactiveMP.RandomVariable, keyname::Symbol) = "ReactiveMP.RandomVariable(label=$(string(z.label)))"

# ╔═╡ f157ba01-48d8-4762-8a2c-5a2a9140b3b2
function to_observability_string(z::Vector{<:ReactiveMP.AbstractMessage}, keyname::Symbol)
	"Vector of $(length(z)) messages."
	# of types $(join(unique(typeof.(z)), ", "))."
end

# ╔═╡ 4f21d86a-628b-45e6-aae3-db1789e30552
to_observability_string(z::Union{
	ReactiveMP.MessageProductContext,
	ReactiveMP.MessageMapping,
	RxInfer.ProbabilisticModel,
}, keyname::Symbol) = "<omitted>"

# ╔═╡ 8361472e-6283-4dac-996b-6c99579bc1d4
begin
	to_observability_string
	
	@generated function struct_to_simple_dict(x)
		fff = fieldnames(x)
		Expr(
			:tuple,
			(
				Expr(Symbol("="), f, :(to_observability_string(getfield(x, $(QuoteNode(f))), $(QuoteNode(f)))))
				for f in fff
			)...
		)
		# quote
		# 	d = Dict{String,String}(
		# 	 # "type" => $(string(x)),
		# 	)
		# 	for f in $(fff)
		# 		d[string(f)] = to_observability_string(getfield(x, f), f)
		# 	end
		# 	d
		# end
	end
end

# ╔═╡ 0f0b0b6f-ed43-4666-b8d8-1debc54cedf6
fieldnames(ReactiveMP.RandomVariable)

# ╔═╡ cebb88a3-6d92-4ef8-afd9-6f9a85ba4f83
time_ns_drift = let
	now_real = round(UInt64, Dates.datetime2unix(Dates.now()) * 1e9)
	now_measured = time_ns()
	now_real - now_measured
end

# ╔═╡ bfa274ff-75dd-4e2e-9022-a3c3f9185ce2
ns_string(t::UInt64) = string(t + time_ns_drift; base=10)

# ╔═╡ 0eaeaf2e-c406-4bd5-bcc8-23bf41d8b739
uuid_to_32bit(u) = replace(string(u), "-" => "")

# ╔═╡ 28df460d-bd14-4160-87ec-79894ae89854
const random_trace_id = uuid_to_32bit(uuid4())

# ╔═╡ 325457c3-c099-4457-9895-63579ba17f70
uuid_to_16bit(u) = replace(string(u), "-" => "")[1:16]

# ╔═╡ ac2a7195-cece-44e1-aea7-e788f88f76cb
uuid_to_16bit(uuid4())

# ╔═╡ 5586e6e2-ad07-4717-a26c-099acb896855
uuid_to_32bit(uuid4())

# ╔═╡ 7469b498-996b-48e2-8dd9-b8c18eaafc1d


# ╔═╡ 3aaa375f-a853-4e85-89a6-6be644564f6e
to_otel_value(v::AbstractString)  = Dict("stringValue" => string(v))

# ╔═╡ d01411b2-7315-44dc-bfb5-396cd7f9eaa9
to_otel_value(v::Int64)  = Dict("intValue" => string(v))

# ╔═╡ 68ccdfeb-08aa-4833-a857-0444e72c6842
to_otel_value(v::Float64)  = Dict("doubleValue" => v)

# ╔═╡ 856362f7-d7bc-4e24-ac06-5f1eaa75e38c
to_otel_value(v::Bool) = Dict("boolValue" => v)

# ╔═╡ 6c3b3e74-dad7-48e5-badd-21eff9e16640
to_otel_value(v::Vector)  = Dict("arrayValue" => Dict("values" => map(to_otel_value, v)))

# ╔═╡ 0ee8645e-3c97-4a0b-8d2f-471f8b32ba0d
to_otel_dict(d::AbstractDict) = [
    Dict("key" => k, "value" => to_otel_value(v))
    for (k,v) in d
]

# ╔═╡ 408490d7-dc1f-49ad-b35c-909397b87668
# ╠═╡ disabled = true
#=╠═╡
function to_otld_span(trace1::TracedEvent, trace2::TracedEvent; 
					  parent_id::Union{Nothing,UUID}=nothing)

	original_name = string(typeof(trace1.event).name.name)
	nice_name = replace(original_name, r"(^Before)|(^After)|(Event$)" => "")
	
	Dict(
	    # "name"                 => string(typeof(trace1.event)),
	    "name"                 => nice_name,
		
	    "traceId"              => random_trace_id,
	    "spanId"               => uuid_to_16bit(trace1.event.trace_id),
	    "parentSpanId"         => parent_id === nothing ? "" : uuid_to_16bit(parent_id),
	    "kind"                 => 0,
	    "startTimeUnixNano"    => ns_string(trace1.time_ns),
	    "endTimeUnixNano"      => ns_string(trace2.time_ns),
	    "attributes"           => to_otel_dict(
			struct_to_simple_dict(trace2.event),
		),
	    "status"               => Dict("code" => 1),
	)
end
  ╠═╡ =#

# ╔═╡ 6d5eb541-0e0e-4ccb-ac13-b6409c990d66
#=╠═╡
to_otld_span(found.spans[7]...; parent_id=found.spans[zzz[7]][1].event.trace_id)
  ╠═╡ =#

# ╔═╡ 649a64f5-f697-41ca-98d3-eb8ab94e6025
#=╠═╡
json_spans = map(1:min(lastindex(zzz),10_000)) do i
	p = zzz[i]
	parent_id = p === nothing ? nothing : found.spans[p][1].event.trace_id
	to_otld_span(found.spans[i]...; parent_id)
end
  ╠═╡ =#

# ╔═╡ c77e4434-df63-4aed-a961-254cc5c59c0c
to_otel_value(v::Dict)  = Dict("kvlistValue" => Dict("values" => to_otel_dict(v)))

# ╔═╡ 3e7c8bf0-6438-40cc-bddc-342f52667a29


# ╔═╡ 30ce5dd8-947a-4129-859a-fea1993af116
const resource_attrs = Dict(
    "service.name" =>     "RxInfer",
    "service.version" =>  string(pkgversion(RxInfer)),
)

# ╔═╡ d33734f9-8f0f-4832-bb79-593a7b39dbe2
#=╠═╡
otlp = Dict(
    "resourceSpans" => [
        Dict(
            "resource" => Dict("attributes" => to_otel_dict(resource_attrs)),
            "scopeSpans" => [
                Dict(
                    "scope" => Dict("name" => "julia-manual", "version" => "1.0"),
                    "spans" => json_spans,
                ),
            ],
        ),
    ],
    "resourceLogs" => [
        Dict(
            "resource" => Dict("attributes" => resource_attrs),
            "scopeLogs" => [
                Dict(
                    "scope" => Dict("name" => "julia-manual", "version" => "1.0"),
                    "logRecords" => [
						# log_record
					],
                ),
            ],
        ),
    ],
)
  ╠═╡ =#

# ╔═╡ b4802d6d-024c-41a0-9cbf-2670fa04864c
#=╠═╡
result = sprint() do io
	JSON3.write(io, otlp)
end
  ╠═╡ =#

# ╔═╡ 298fc244-24fd-4dc0-a140-a602e7170e10
#=╠═╡
"$(round(length(result) / 1e6, digits=2)) MB" |> Text
  ╠═╡ =#

# ╔═╡ e93851c8-15db-4f72-abee-34f23837d1e8
#=╠═╡
PlutoUI.DownloadButton(result, "result.json")
  ╠═╡ =#

# ╔═╡ 3498ce24-5d99-454c-ae1b-9cf7c2249b1e
md"""
# Perfetto export
"""

# ╔═╡ 0676725a-907e-4e94-a77d-708da32524f2
after_marginal_events

# ╔═╡ 9e475912-b6d7-4086-a092-59787f014917
supertype(after_marginal_events[1].event |> typeof)

# ╔═╡ ccadfd66-7274-4ceb-bfa6-c5d53f335a32
@generated function event_phase(t::ReactiveMP.Event)
	name = t.name.name
	startswith(string(name), "Before") ? "B" :
	startswith(string(name), "After") ? "E" :
		"X"
end

# ╔═╡ 3afd1265-bd5d-40cc-ae47-617930b189c9
function to_perfetto(te::TracedEvent; time_delta::Int64=0)
	e = te.event
	t = typeof(e).name.name

	original_name = string(t)
	nice_name = replace(original_name, r"(^Before)|(^After)|(Event$)" => "")

	ph = event_phase(e)
	
	(
	    pid = 1,
	    tid = 1,
	    name = nice_name,
	    # cat = "default",
	    ph = ph,
	    ts = Float64(
	        Int64(te.time_ns) +
	            time_delta +
	            (ph == "E" ? 1 : 0) # add 1 ns to End events because some recorded events are 0ns long, which is otherwise not supported in perfetto.
	    ) / 1000,
	    id = string(get_trace_id(e)),
	    args = struct_to_simple_dict(e),
	)
end

# ╔═╡ d3bbed88-8539-4e88-9e7a-58f788e40864
to_perfetto(after_marginal_events[711])

# ╔═╡ bbffd54b-4511-4f38-970d-eb40a9bd036d


# ╔═╡ 6f45ef45-5f17-4d59-8f1b-e5f964ea0f36
aa = after_marginal_events[1].time_ns

# ╔═╡ 27cc69f4-cd23-4493-bda7-eabe12bfb2a3
bb = after_marginal_events[3].time_ns

# ╔═╡ 07488896-bb4a-44a3-afdb-b0fa94a952b1
Int64(bb - aa) / 1e3

# ╔═╡ c53048fe-105a-4841-be9d-0173c711916c
perfetto_traces = let
	time_delta = -Int64(after_marginal_events[1].time_ns)
	to_perfetto.(after_marginal_events; time_delta=time_delta)
end

# ╔═╡ 2aaaddba-7bc5-41d3-b92c-2c5b3f9fe25c


# ╔═╡ 762fdff8-ada8-42f0-9505-d0a0b86f7014
perfetto = Dict(
	"traceEvents" => perfetto_traces,
	"metadata" => Dict(
		
		"clock-domain" => "MONO",
		"trace-capture-datetime" => 
		"command_line" => "RxInferBoard",
		"higres-ticks" => true,
	)
)

# ╔═╡ 673f532b-1aae-4196-824f-c51ad4aef2e2
result_perfetto = sprint() do io
	JSON3.write(io, perfetto)
end

# ╔═╡ 265bd35e-2832-4a80-830b-6029ee4f2688
"$(round(length(result_perfetto) / 1e6, digits=2)) MB" |> Text

# ╔═╡ 99c68335-8a0f-440e-adaf-c23ac4f1a36f


# ╔═╡ cb5b75b5-2224-45cc-98ae-b0b5f9c0cf64
PlutoUI.DownloadButton(result_perfetto, "result_perfetto.json")

# ╔═╡ d8cb0743-3f42-4d3c-951b-ed22b3fca916
filter(after_marginal_events) do te
	get_trace_id(te.event) == Base.UUID("bab13dad-e17a-4638-b190-4dbadbea4736")
end

# ╔═╡ 91727b0d-8cf2-42c3-b1fc-c9af91b48863
md"""
# Open Perfetto from Julia
Uncomment these cells to try it:
"""

# ╔═╡ bf6b7471-7ccb-4884-bc4e-a82eb18dcd46
# open_with_perfetto(result_perfetto)

# ╔═╡ 9f438e0a-6b2f-4d06-a66c-517d5ba85c16
# view_with_perfetto(result_perfetto)

# ╔═╡ a26e8294-fb5d-4b4c-a9c1-b8780d809a37


# ╔═╡ fad878e1-bf48-4d91-8f1b-3f9c93a69463
function view_with_perfetto(perfetto_json_contents; name = "RxInfer trace")
	b64 = Base64.base64encode(perfetto_json_contents)

	id = String(rand('a':'z', 10))

	file = """
		<div style="width: 100%; height: clamp(650px, 90vh, 1000px);">
	<iframe id=$id src="https://ui.perfetto.dev"
	  style="width:100%;height:100%;border:7px solid yellow;border-radius: 12px;"></iframe>
	<script>
	const b64 = "$(b64)";
	const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
	const iframe = document.getElementById('$(id)');
	
	// Keep sending PING until Perfetto replies with PONG
	const interval = setInterval(() => {
	  iframe.contentWindow.postMessage('PING', 'https://ui.perfetto.dev');
	}, 50);
	
	window.addEventListener('message', (e) => {
	  if (e.data !== 'PONG') return;
	  clearInterval(interval);
	  iframe.contentWindow.postMessage({
	    perfetto: { 
	      buffer: bytes.buffer, 
	      title: "$(name)",
	    }
	  }, 'https://ui.perfetto.dev');
	});
	</script>
	</div>
	"""

	HTML(file) |> PlutoUI.WideCell
end

# ╔═╡ 5287b97c-2148-4186-a8d5-127a396132f9
function open_with_perfetto(perfetto_json_contents; name = "RxInfer trace")
	b64 = Base64.base64encode(perfetto_json_contents)

	file = """
	<!DOCTYPE html><html><body style="margin:0">
	<iframe id="pf" src="https://ui.perfetto.dev"
	  style="width:100vw;height:100vh;border:none;position:fixed;top:0;left:0"></iframe>
		
		<div id="overlay" style="
	  position:fixed;top:0;left:0;width:100vw;height:100vh;
	  background:rgba(255,255,255,0.5);
	  display:flex;align-items:center;justify-content:center;
	  transition:opacity 0.4s ease;
	"><span style="font:bold 3rem system-ui;white-space:nowrap">Loading...</span></div>
	
	<script>
	const b64 = "$(b64)";
	const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
	const iframe = document.getElementById('pf');
	const overlay = document.getElementById('overlay');

	
	// Keep sending PING until Perfetto replies with PONG
	const interval = setInterval(() => {
	  iframe.contentWindow.postMessage('PING', 'https://ui.perfetto.dev');
	}, 50);
	
	window.addEventListener('message', (e) => {
	  if (e.data !== 'PONG') return;
	  clearInterval(interval);
	  iframe.contentWindow.postMessage({
	    perfetto: { 
	      buffer: bytes.buffer, 
	      title: "$(name)",
	    }
	  }, 'https://ui.perfetto.dev');
		
	  overlay.style.opacity = '0';
	  setTimeout(() => overlay.remove(), 400);
	});
	</script>
	</body></html>
	"""

	filename = tempname(cleanup=false) * ".html"
	write(filename, file)

	if Sys.isapple()
        run(`open $filename`)
    elseif Sys.iswindows()
        run(`cmd /c start "" $filename`)
    elseif Sys.islinux()
        run(`xdg-open $filename`)
    else
        @info("Open this in your browser: $filename")
    end
end

# ╔═╡ Cell order:
# ╟─18d21cb8-e14f-4cbd-b679-7c696eac8b21
# ╟─1e8a46ca-268c-11f1-8687-1362ac8b895a
# ╠═c64de8a5-71fc-4a31-8fe7-953b73158eda
# ╠═28c0af84-e133-4e70-8728-4846c80d1add
# ╠═c5bf6e4a-bd50-4d45-9e13-b2f39c730e08
# ╟─c29b75a6-fdac-45e7-a69f-583dbad8c60d
# ╠═45f9d11c-6c53-4294-9e64-31b10bb7a7f8
# ╠═2b449df0-414e-41ea-889a-cdb399bbb61c
# ╠═aa0c87e6-2a10-48e9-ae1e-f92618cf90b8
# ╠═f63901f0-edd6-40f0-9537-9e37f2d38c4b
# ╠═61fcb0f3-4ff0-4cbb-bb72-8b6f22457902
# ╟─4cb2aaca-01eb-4ab5-9223-d59514774d18
# ╠═e01f8c2a-2b13-403a-bd0d-7751567383e7
# ╠═ac9085bf-4463-4f71-ae5e-8f4b59e056c9
# ╟─66399916-d2d5-4e18-8b84-ef1e10074b49
# ╠═d13c9fbb-0771-4d8e-82ae-563875cad3ce
# ╠═4dc178b5-d0a0-46ac-92b8-cbe45e7d0b3a
# ╠═b8fdcbe7-2ed7-4cd1-b215-cb0905b7efe2
# ╠═d139d769-d411-4718-ab0a-cb6f176afd9d
# ╠═2542529b-f49c-4167-a6a3-8a099ee942d0
# ╠═98d10049-164f-44b4-8c6c-6a1714aa0a3d
# ╟─d12d24be-938c-4fd6-918d-955427ab8b50
# ╠═cf550cc4-5dd8-4340-875c-2032441ce765
# ╠═04407b00-3308-4e35-b144-4dac46afc252
# ╟─f479106f-d65c-479a-8ef6-3fe8dbe391b7
# ╠═866fad2d-0d4d-4ff9-a355-4c0c1cbbf6bb
# ╠═252de4a1-24d7-4c84-8196-44e0fb50b37b
# ╠═28df460d-bd14-4160-87ec-79894ae89854
# ╠═408490d7-dc1f-49ad-b35c-909397b87668
# ╠═8361472e-6283-4dac-996b-6c99579bc1d4
# ╠═1afa52be-6a37-4049-b0f9-e9d0d720f31f
# ╠═0747b0f9-1081-4bb9-8193-6d3d2c59cda4
# ╠═299a9c0d-1ef6-421d-ba97-317a913b1940
# ╠═2c049a69-24d1-43ab-bd18-4ab9828628df
# ╠═57251562-4f84-4dd5-b0af-9000983366d9
# ╠═31455dd1-1755-46ed-995a-00c17d1653b2
# ╠═f157ba01-48d8-4762-8a2c-5a2a9140b3b2
# ╠═4f21d86a-628b-45e6-aae3-db1789e30552
# ╠═6d5eb541-0e0e-4ccb-ac13-b6409c990d66
# ╠═0f0b0b6f-ed43-4666-b8d8-1debc54cedf6
# ╠═cebb88a3-6d92-4ef8-afd9-6f9a85ba4f83
# ╠═649a64f5-f697-41ca-98d3-eb8ab94e6025
# ╠═bfa274ff-75dd-4e2e-9022-a3c3f9185ce2
# ╠═8e8141e0-2b74-4813-ac36-2dcb6e86df47
# ╠═0eaeaf2e-c406-4bd5-bcc8-23bf41d8b739
# ╠═325457c3-c099-4457-9895-63579ba17f70
# ╠═ac2a7195-cece-44e1-aea7-e788f88f76cb
# ╠═5586e6e2-ad07-4717-a26c-099acb896855
# ╟─7469b498-996b-48e2-8dd9-b8c18eaafc1d
# ╟─3aaa375f-a853-4e85-89a6-6be644564f6e
# ╟─d01411b2-7315-44dc-bfb5-396cd7f9eaa9
# ╟─68ccdfeb-08aa-4833-a857-0444e72c6842
# ╟─856362f7-d7bc-4e24-ac06-5f1eaa75e38c
# ╟─6c3b3e74-dad7-48e5-badd-21eff9e16640
# ╟─c77e4434-df63-4aed-a961-254cc5c59c0c
# ╟─0ee8645e-3c97-4a0b-8d2f-471f8b32ba0d
# ╟─3e7c8bf0-6438-40cc-bddc-342f52667a29
# ╠═30ce5dd8-947a-4129-859a-fea1993af116
# ╠═d33734f9-8f0f-4832-bb79-593a7b39dbe2
# ╠═b4802d6d-024c-41a0-9cbf-2670fa04864c
# ╠═298fc244-24fd-4dc0-a140-a602e7170e10
# ╠═93480c78-5e9c-4398-b2e4-414249553545
# ╠═e93851c8-15db-4f72-abee-34f23837d1e8
# ╟─3498ce24-5d99-454c-ae1b-9cf7c2249b1e
# ╠═0676725a-907e-4e94-a77d-708da32524f2
# ╠═9e475912-b6d7-4086-a092-59787f014917
# ╠═ccadfd66-7274-4ceb-bfa6-c5d53f335a32
# ╠═d3bbed88-8539-4e88-9e7a-58f788e40864
# ╠═3afd1265-bd5d-40cc-ae47-617930b189c9
# ╠═bbffd54b-4511-4f38-970d-eb40a9bd036d
# ╠═6f45ef45-5f17-4d59-8f1b-e5f964ea0f36
# ╠═27cc69f4-cd23-4493-bda7-eabe12bfb2a3
# ╠═07488896-bb4a-44a3-afdb-b0fa94a952b1
# ╠═c53048fe-105a-4841-be9d-0173c711916c
# ╠═2aaaddba-7bc5-41d3-b92c-2c5b3f9fe25c
# ╠═7883817d-d74d-40c8-b1a2-08904fdc0f9c
# ╠═19609cfa-f124-4060-976f-40102d7cc45d
# ╠═762fdff8-ada8-42f0-9505-d0a0b86f7014
# ╠═673f532b-1aae-4196-824f-c51ad4aef2e2
# ╠═265bd35e-2832-4a80-830b-6029ee4f2688
# ╠═99c68335-8a0f-440e-adaf-c23ac4f1a36f
# ╠═cb5b75b5-2224-45cc-98ae-b0b5f9c0cf64
# ╠═d8cb0743-3f42-4d3c-951b-ed22b3fca916
# ╠═42f96fd4-8c47-4c41-895d-aeecbb3aa19a
# ╟─91727b0d-8cf2-42c3-b1fc-c9af91b48863
# ╠═bf6b7471-7ccb-4884-bc4e-a82eb18dcd46
# ╠═9f438e0a-6b2f-4d06-a66c-517d5ba85c16
# ╟─a26e8294-fb5d-4b4c-a9c1-b8780d809a37
# ╠═fad878e1-bf48-4d91-8f1b-3f9c93a69463
# ╠═5287b97c-2148-4186-a8d5-127a396132f9
