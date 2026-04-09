export perfetto_view, perfetto_open

import JSON
import Base64
import Dates

function _traces_to_perfetto_json(traces::Vector{TracedEvent})
    time_delta = isempty(traces) ? 0 : -Int64(traces[1].time_ns) # Shift all timestamps so the first event starts at t=0

    # Convert traces to dict-like structures in Perfetto format
    perfetto_events = _to_perfetto_event.(traces; time_delta)

    return JSON.json(
        Dict(
            "traceEvents" => perfetto_events,
            "metadata" => Dict(
                "clock-domain" => "MONO",
                "command_line" => "RxInfer",
                "higres-ticks" => true,
            ),
        ),
    )
end

function _to_perfetto_event(te::TracedEvent; time_delta::Int64 = 0)
    e = te.event
    original_name = string(typeof(e).name.name)
    nice_name = replace(original_name, r"(^Before)|(^After)|(Event$)" => "")
    ph = _event_phase(e)
    (
        pid = 1,
        tid = 1,
        name = nice_name,
        # cat = "default",
        ph = ph,
        # Timestamps are in microseconds for Perfetto.
        # Add 1 ns to End events so zero-duration spans are visible.
        ts = Float64(Int64(te.time_ns) + time_delta + (ph == "E" ? 1 : 0)) /
             1000,
        # id = string(_get_trace_id(e)),
        args = _struct_to_simple_dict(e),
    )
end

@generated function _struct_to_simple_dict(x)
    struct_field_names = fieldnames(x)
    # This is a very optimised function because this is performance critical.
    # It is a generated function that compiles down to a literal named tuple with field names matching the struct fields.
    if isempty(struct_field_names)
        :((;))
    else
        Expr(
            :tuple,
            (
                Expr(
                    Symbol("="),
                    field_name,
                    :(_to_observability_string(
                        getfield(x, $(QuoteNode(field_name))),
                        $(QuoteNode(field_name)),
                    )),
                ) for field_name in struct_field_names# if field_name !== :trace_id
            )...,
        )
    end

    ## Less optimized version using Dict:
    # quote
    # 	d = Dict{String,String}(
    # 	 # "type" => $(string(x)),
    # 	)
    # 	for f in $(struct_field_names)
    # 		d[string(f)] = to_observability_string(getfield(x, f), f)
    # 	end
    # 	d
    # end
end

# _to_observability_string converts a field of an event object to a string to view in Perfetto. This should be high performance and give useful info.
# The default is to call `string(...)`, but uou can add methods for fields to override their behaviour.

# Default fallback
function _to_observability_string(z::Any, _keyname::Symbol)
    string(z)

    ## Uncomment this to find fields that take a long time to convert to string:
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
end

_to_observability_string(z::ReactiveMP.RandomVariable, _keyname::Symbol) = "ReactiveMP.RandomVariable(label=$(z.label))"

_to_observability_string(z::Vector{<:ReactiveMP.AbstractMessage}, _keyname::Symbol) = "Vector of $(length(z)) messages."

_to_observability_string(::ReactiveMP.MessageProductContext, _keyname::Symbol) = "<omitted>"
_to_observability_string(::ReactiveMP.MessageMapping, _keyname::Symbol) = "<omitted>"
_to_observability_string(::ProbabilisticModel, _keyname::Symbol) = "<omitted>"

"""
    _time_ns_to_datetime(t::UInt64) -> Dates.DateTime

Converts a `time_ns()` timestamp to a wall-clock `DateTime` by computing the
drift between Julia's monotonic clock and the Unix epoch at the moment of the call.
"""
function _time_ns_to_datetime(t::UInt64)::Dates.DateTime
    now_real_ns = round(Int64, Dates.datetime2unix(Dates.now()) * 1e9)
    drift = now_real_ns - Int64(time_ns())
    return Dates.unix2datetime((Int64(t) + drift) / 1e9)
end

function _default_trace_name(traces::Vector{TracedEvent})
    isempty(traces) && return "RxInfer trace"
    return "$(Dates.Time(_time_ns_to_datetime(traces[1].time_ns))) RxInfer trace"
end

@generated function _event_phase(e::ReactiveMP.Event)
    name = string(e.name.name)
    startswith(name, "Before") && return "B"
    startswith(name, "After") && return "E"
    return "X"
end

# @generated function _get_trace_id(event::ReactiveMP.Event)
# 	if hasfield(event, :trace_id)
# 		:(event.trace_id)
# 	end
# end

### ---- Display functionality

"""
    PerfettoDisplay

Returned by [`perfetto_view`](@ref). Renders as an embedded [Perfetto](https://ui.perfetto.dev)
trace viewer when displayed in a Pluto or Jupyter notebook cell.
"""
struct PerfettoDisplay
    html::String
end

function Base.show(io::IO, ::MIME"text/html", p::PerfettoDisplay)
    print(io, p.html)
end

Base.show(io::IO, ::PerfettoDisplay) = print(
    io,
    "PerfettoDisplay (render in a Pluto or Jupyter notebook to see the interactive trace)",
)

"""
    perfetto_view(traces::Vector{TracedEvent}; name = "\$(Time(now())) RxInfer trace")

Converts a vector of [`TracedEvent`](@ref)s to an embedded [Perfetto](https://ui.perfetto.dev)
trace viewer. Returns a [`PerfettoDisplay`](@ref) that renders as an interactive trace when
displayed in a Pluto or Jupyter notebook cell.

See also: [`perfetto_open`](@ref), [`RxInferTraceCallbacks`](@ref).


# Example
```julia
result = infer(model = my_model(), data = my_data, trace = true)
traces = RxInfer.tracedevents(result.model.metadata[:trace])
perfetto_view(traces)   # display in a notebook cell
```


_Pluto tip: combine this with `PlutoUI.WideCell` for a bigger view, so `perfetto_view(traces) |> WideCell`._
"""
function perfetto_view(
    traces::Vector{TracedEvent}; name::String = _default_trace_name(traces)
)
    json_contents = _traces_to_perfetto_json(traces)
    b64 = Base64.base64encode(json_contents)
    id = String(rand('a':'z', 10))
    html = """
        <div style="width: 100%; height: clamp(650px, 90vh, 1000px);">
        <iframe id="$id" src="https://ui.perfetto.dev"
          style="width:100%;height:100%;border:7px solid yellow;border-radius: 12px;"></iframe>
        <script>
        const b64 = "$b64";
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const iframe = document.getElementById('$id');

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
        </div>"""
    return PerfettoDisplay(html)
end

"""
    perfetto_open(traces::Vector{TracedEvent}; name = "\$(Time(now())) RxInfer trace")

Converts a vector of [`TracedEvent`](@ref)s to Perfetto JSON and opens the result in your
default web browser using the [Perfetto](https://ui.perfetto.dev) trace viewer.

Returns the path to the temporary HTML file that was opened.

See also: [`perfetto_view`](@ref), [`RxInferTraceCallbacks`](@ref).

# Example
```julia
result = infer(model = my_model(), data = my_data, trace = true)
traces = RxInfer.tracedevents(result.model.metadata[:trace])
perfetto_open(traces)
```
"""
function perfetto_open(
    traces::Vector{TracedEvent}; name::String = _default_trace_name(traces)
)
    json_contents = _traces_to_perfetto_json(traces)
    b64 = Base64.base64encode(json_contents)
    html = """<!DOCTYPE html><html><body style="margin:0">
        <iframe id="pf" src="https://ui.perfetto.dev"
          style="width:100vw;height:100vh;border:none;position:fixed;top:0;left:0"></iframe>
        <div id="overlay" style="
          position:fixed;top:0;left:0;width:100vw;height:100vh;
          background:rgba(255,255,255,0.5);
          display:flex;align-items:center;justify-content:center;
          transition:opacity 0.4s ease;">
          <span style="font:bold 3rem system-ui;white-space:nowrap">Loading...</span>
        </div>
        <script>
        const b64 = "$b64";
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const iframe = document.getElementById('pf');
        const overlay = document.getElementById('overlay');

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
        </body></html>"""
    filename = tempname(; cleanup = false) * ".html"
    write(filename, html)
    if Sys.isapple()
        run(`open $filename`)
    elseif Sys.iswindows()
        run(`cmd /c start "" $filename`)
    elseif Sys.islinux()
        run(`xdg-open $filename`)
    else
        @info "Open this in your browser: $filename"
    end
    return filename
end
