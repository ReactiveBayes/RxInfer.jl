export BeforeModelCreationEvent, AfterModelCreationEvent
export BeforeInferenceEvent, AfterInferenceEvent
export BeforeIterationEvent, AfterIterationEvent
export BeforeDataUpdateEvent, AfterDataUpdateEvent
export OnMarginalUpdateEvent
export BeforeAutostartEvent, AfterAutostartEvent

import ReactiveMP: Event, event_name, generate_span_id

## RxInfer-level callback event types
## These events subtype `ReactiveMP.Event{E}` and carry relevant data as fields.

"""
    BeforeModelCreationEvent{S} <: ReactiveMP.Event{:before_model_creation}

Fires right before the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `span_id`: an identifier shared with the corresponding [`AfterModelCreationEvent`](@ref)

See also: [`AfterModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeModelCreationEvent{S} <: Event{:before_model_creation}
    span_id::S
end

"""
    AfterModelCreationEvent{M, S} <: ReactiveMP.Event{:after_model_creation}

Fires right after the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `model::M`: the created [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeModelCreationEvent`](@ref)

See also: [`BeforeModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterModelCreationEvent{M, S} <: Event{:after_model_creation}
    model::M
    span_id::S
end

"""
    BeforeInferenceEvent{M, S} <: ReactiveMP.Event{:before_inference}

Fires right before the inference procedure starts (after model creation and subscription setup).

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`AfterInferenceEvent`](@ref)

See also: [`AfterInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeInferenceEvent{M, S} <: Event{:before_inference}
    model::M
    span_id::S
end

"""
    AfterInferenceEvent{M, S} <: ReactiveMP.Event{:after_inference}

Fires right after the inference procedure completes.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeInferenceEvent`](@ref)

See also: [`BeforeInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterInferenceEvent{M, S} <: Event{:after_inference}
    model::M
    span_id::S
end

"""
    BeforeIterationEvent{M, S} <: ReactiveMP.Event{:before_iteration}

Fires right before each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `span_id`: an identifier shared with the corresponding [`AfterIterationEvent`](@ref)

See also: [`AfterIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct BeforeIterationEvent{M, S} <: Event{:before_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    span_id::S
end

BeforeIterationEvent(model::M, iteration::Int, span_id) where {M} = BeforeIterationEvent(
    model, iteration, false, span_id
)

"""
    AfterIterationEvent{M, S} <: ReactiveMP.Event{:after_iteration}

Fires right after each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `span_id`: an identifier shared with the corresponding [`BeforeIterationEvent`](@ref)

See also: [`BeforeIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct AfterIterationEvent{M, S} <: Event{:after_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    span_id::S
end

AfterIterationEvent(model::M, iteration::Int, span_id) where {M} = AfterIterationEvent(
    model, iteration, false, span_id
)

"""
    BeforeDataUpdateEvent{M, D, S} <: ReactiveMP.Event{:before_data_update}

Fires right before updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data being used for the update
- `span_id`: an identifier shared with the corresponding [`AfterDataUpdateEvent`](@ref)

See also: [`AfterDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeDataUpdateEvent{M, D, S} <: Event{:before_data_update}
    model::M
    data::D
    span_id::S
end

"""
    AfterDataUpdateEvent{M, D, S} <: ReactiveMP.Event{:after_data_update}

Fires right after updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data that was used for the update
- `span_id`: an identifier shared with the corresponding [`BeforeDataUpdateEvent`](@ref)

See also: [`BeforeDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterDataUpdateEvent{M, D, S} <: Event{:after_data_update}
    model::M
    data::D
    span_id::S
end

"""
    OnMarginalUpdateEvent{M, U} <: ReactiveMP.Event{:on_marginal_update}

Fires each time a marginal posterior for a variable is updated during inference.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `variable_name::Symbol`: the name of the variable whose marginal was updated
- `update::U`: the updated marginal value

See also: [Callbacks](@ref manual-inference-callbacks)
"""
struct OnMarginalUpdateEvent{M, U} <: Event{:on_marginal_update}
    model::M
    variable_name::Symbol
    update::U
end

"""
    BeforeAutostartEvent{E, S} <: ReactiveMP.Event{:before_autostart}

Fires right before `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`AfterAutostartEvent`](@ref)

See also: [`AfterAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeAutostartEvent{E, S} <: Event{:before_autostart}
    engine::E
    span_id::S
end

"""
    AfterAutostartEvent{E, S} <: ReactiveMP.Event{:after_autostart}

Fires right after `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeAutostartEvent`](@ref)

See also: [`BeforeAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterAutostartEvent{E, S} <: Event{:after_autostart}
    engine::E
    span_id::S
end
# ─────────────────────────────────────────────────────────────────────────────
# 1. Graph indexing
# ─────────────────────────────────────────────────────────────────────────────

"""
    GraphIndex

Stable, human-readable identifiers for every factor node, variable, and edge
in a ReactiveMP factor graph. Built once per `infer(...)` run via
[`build_index`](@ref).

# Fields
- `node_ids   :: IdDict{Any,String}`        — `factor node → "N1"`
- `var_ids    :: IdDict{Any,String}`        — `variable    → "V1"`
- `edge_ids   :: Dict{(String,String),String}` — `(NodeID, VarID) → "E1"`
- `var_to_nodes :: IdDict{Any,Vector{Any}}` — reverse adjacency for events
  that only carry a `variable` reference
- `node_form  :: IdDict{Any,String}`        — `node → "Normal"` etc.
- `var_label  :: IdDict{Any,String}`        — `variable → "μ"` etc.
- `nodes`, `vars`                           — insertion-ordered raw objects
"""
struct GraphIndex
    node_ids     :: IdDict{Any, String}
    var_ids      :: IdDict{Any, String}
    edge_ids     :: Dict{Tuple{String, String}, String}
    var_to_nodes :: IdDict{Any, Vector{Any}}
    node_form    :: IdDict{Any, String}
    var_label    :: IdDict{Any, String}
    nodes        :: Vector{Any}
    vars         :: Vector{Any}
end

# Internal helpers — defensive accessors so we degrade gracefully if RxInfer
# renames an API in a future release.

_safe_unwrap_node(nd) =
    try
        RxInfer.getextra(nd, RxInfer.ReactiveMPExtraFactorNodeKey)
    catch
        nd
    end

_safe_unwrap_var(nd) =
    try
        RxInfer.getvariable(nd)
    catch
        nd
    end

_safe_form(node) =
    try
        string(ReactiveMP.functionalform(node))
    catch
        string(typeof(node).name.name)
    end

function _safe_var_label(v)
    try
        return string(v.label.name)
    catch
        try
            return string(v.name)
        catch
            return string(typeof(v).name.name)
        end
    end
end

_safe_interfaces(node) =
    try
        collect(ReactiveMP.getinterfaces(node))
    catch
        Any[]
    end

"""
    build_index(model) -> GraphIndex

Walk the post-`infer` model and assign every factor node, variable and edge
a stable string ID (`N1`, `V1`, `E1`, …). The returned [`GraphIndex`](@ref)
is the foundation every other tool in this file builds on.
"""
function build_index(model)::GraphIndex
    node_ids     = IdDict{Any, String}()
    var_ids      = IdDict{Any, String}()
    edge_ids     = Dict{Tuple{String, String}, String}()
    var_to_nodes = IdDict{Any, Vector{Any}}()
    node_form    = IdDict{Any, String}()
    var_label    = IdDict{Any, String}()
    nodes        = Any[]
    vars         = Any[]

    var_counter = 0
    for getter in
        (RxInfer.getrandomvars, RxInfer.getdatavars, RxInfer.getconstantvars)
        local nds
        try
            nds = getter(model)
        catch
            continue
        end
        for nd in nds
            v = _safe_unwrap_var(nd)
            if !haskey(var_ids, v)
                var_counter  += 1
                var_ids[v]   = "V$var_counter"
                var_label[v] = _safe_var_label(v)
                push!(vars, v)
            end
        end
    end

    node_counter = 0
    local factor_nds
    try
        factor_nds = RxInfer.getfactornodes(model)
    catch
        factor_nds = Any[]
    end
    for nd in factor_nds
        n = _safe_unwrap_node(nd)
        if !haskey(node_ids, n)
            node_counter += 1
            node_ids[n] = "N$node_counter"
            node_form[n] = _safe_form(n)
            push!(nodes, n)
        end
    end

    edge_counter = 0
    for n in nodes
        nid = node_ids[n]
        for iface in _safe_interfaces(n)
            v = nothing
            try
                v = ReactiveMP.getvariable(iface)
            catch
                continue
            end
            v === nothing && continue
            if !haskey(var_ids, v)
                var_counter  += 1
                var_ids[v]   = "V$var_counter"
                var_label[v] = _safe_var_label(v)
                push!(vars, v)
            end
            vid = var_ids[v]
            key = (nid, vid)
            if !haskey(edge_ids, key)
                edge_counter += 1
                edge_ids[key] = "E$edge_counter"
            end
            push!(get!(var_to_nodes, v, Any[]), n)
        end
    end

    return GraphIndex(
        node_ids,
        var_ids,
        edge_ids,
        var_to_nodes,
        node_form,
        var_label,
        nodes,
        vars,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Event enrichment
# ─────────────────────────────────────────────────────────────────────────────

"""
    EnrichedEvent

A single ReactiveMP callback event annotated with `(idx, iteration, kind,
phase, node_id, edge_ids, variable_id, summary, flags, raw)`. Produced by
[`enrich`](@ref).

`flags` is a `Vector{Symbol}` drawn from
`(:nan, :inf, :variance_collapse, :variance_explode, :vague_late, :no_after)`.
See `event_trace_gui.md` for what each flag means and how it's detected.
"""
struct EnrichedEvent
    idx         :: Int
    iteration   :: Int
    kind        :: Symbol      # :rule | :prod2 | :prodN | :marginal | :other
    phase       :: Symbol      # :before | :after | :n_a
    node_id     :: Union{String, Nothing}
    edge_ids    :: Vector{String}
    variable_id :: Union{String, Nothing}
    summary     :: NamedTuple
    flags       :: Vector{Symbol}
    raw         :: Any
end

function _extract_dist(x)
    try
        return ReactiveMP.getdata(x)
    catch
    end
    for f in (:data, :result)
        if hasproperty(x, f)
            return getfield(x, f)
        end
    end
    return x
end

function _safe_stats(d)
    has_nan = false;
    has_inf = false;
    m = NaN;
    v = NaN;
    is_vague = false
    try
        m = mean(d)
        if m isa AbstractArray
            has_nan = any(isnan, m);
            has_inf = any(isinf, m);
            m = first(m)
        else
            has_nan = isnan(m);
            has_inf = isinf(m)
        end
    catch
    end
    try
        v = var(d)
        if v isa AbstractArray
            has_nan |= any(isnan, v);
            has_inf |= any(isinf, v);
            v = first(v)
        else
            has_nan |= isnan(v);
            has_inf |= isinf(v)
        end
        if isfinite(v) && v > 1e10
            is_vague = true
        end
    catch
    end
    return (
        mean = m,
        var = v,
        has_nan = has_nan,
        has_inf = has_inf,
        is_vague = is_vague,
        dist_type = string(typeof(d).name.name),
    )
end

function _event_kind_phase(ev)
    n = ReactiveMP.event_name(ev)
    s = string(n)
    phase =
        startswith(s, "before_") ? :before :
        startswith(s, "after_")  ? :after  : :n_a
    kind = if occursin("message_rule", s)
        :rule
    elseif occursin("product_of_two", s)
        :prod2
    elseif occursin("product_of_messages", s)
        :prodN
    elseif occursin("marginal_computation", s)
        :marginal
    else
        :other
    end
    return kind, phase
end

function _node_from_event(ev, idx::GraphIndex)
    if hasproperty(ev, :mapping)
        try
            n = ev.mapping.factornode
            return get(idx.node_ids, n, nothing), [n]
        catch
        end
    end
    return nothing, Any[]
end

function _var_from_event(ev, idx::GraphIndex)
    if hasproperty(ev, :variable)
        v = ev.variable
        return get(idx.var_ids, v, nothing), v
    end
    return nothing, nothing
end

"""
    enrich(trace, index::GraphIndex) -> Vector{EnrichedEvent}

Walk every event in `trace` (typically `results.model.metadata[:trace]`),
resolve its node/edge/variable references against `index`, compute summary
statistics for the resulting distribution, and attach anomaly flags.

Single linear pass; safe to call on the full event stream.
"""
function enrich(trace, idx::GraphIndex)::Vector{EnrichedEvent}
    out = EnrichedEvent[]
    raw_events = RxInfer.tracedevents(trace)
    iteration = 1
    seen_var_this_iter = Set{Any}()

    for (i, te) in enumerate(raw_events)
        ev = te.event
        kind, phase = _event_kind_phase(ev)

        if kind == :marginal && phase == :after && hasproperty(ev, :variable)
            if ev.variable in seen_var_this_iter
                iteration += 1
                empty!(seen_var_this_iter)
            end
            push!(seen_var_this_iter, ev.variable)
        end

        node_id, nodes_used = _node_from_event(ev, idx)
        var_id, var_used    = _var_from_event(ev, idx)

        edge_ids = String[]
        if !isempty(nodes_used)
            for n in nodes_used
                nid = get(idx.node_ids, n, nothing)
                nid === nothing && continue
                for iface in _safe_interfaces(n)
                    try
                        v = ReactiveMP.getvariable(iface)
                        vid = get(idx.var_ids, v, nothing)
                        vid === nothing && continue
                        eid = get(idx.edge_ids, (nid, vid), nothing)
                        eid === nothing || push!(edge_ids, eid)
                    catch
                    end
                end
            end
        elseif var_used !== nothing && haskey(idx.var_to_nodes, var_used)
            vid = idx.var_ids[var_used]
            for n in idx.var_to_nodes[var_used]
                nid = get(idx.node_ids, n, nothing)
                nid === nothing && continue
                eid = get(idx.edge_ids, (nid, vid), nothing)
                eid === nothing || push!(edge_ids, eid)
            end
        end

        d = nothing
        if hasproperty(ev, :result)
            d = _extract_dist(ev.result)
        elseif hasproperty(ev, :messages) &&
            ev.messages !== nothing &&
            !isempty(ev.messages)
            d = _extract_dist(first(ev.messages))
        end
        summary = if d === nothing
            (
                mean = NaN,
                var = NaN,
                has_nan = false,
                has_inf = false,
                is_vague = false,
                dist_type = "—",
            )
        else
            _safe_stats(d)
        end

        flags = Symbol[]
        summary.has_nan && push!(flags, :nan)
        summary.has_inf && push!(flags, :inf)
        if summary.is_vague &&
            iteration > 1 &&
            kind == :marginal &&
            phase == :after
            push!(flags, :vague_late)
        end
        if isfinite(summary.var) && summary.var < 1e-10 && summary.var > 0
            push!(flags, :variance_collapse)
        end

        push!(
            out,
            EnrichedEvent(
                i,
                iteration,
                kind,
                phase,
                node_id,
                unique(edge_ids),
                var_id,
                summary,
                flags,
                ev,
            ),
        )
    end

    # variance_explode (compare consecutive events on same edge)
    last_var = Dict{String, Float64}()
    for e in out
        e.kind in (:rule, :marginal) || continue
        e.phase == :after || continue
        isfinite(e.summary.var) || continue
        for eid in e.edge_ids
            if haskey(last_var, eid)
                prev = last_var[eid]
                if prev > 0 && e.summary.var > 10 * prev
                    push!(e.flags, :variance_explode)
                end
            end
            last_var[eid] = e.summary.var
        end
    end

    # no_after pairing check
    pending = Dict{
        Tuple{Symbol, Union{String, Nothing}, Union{String, Nothing}}, Int
    }()
    for e in out
        key = (e.kind, e.node_id, e.variable_id)
        if e.phase == :before
            pending[key] = e.idx
        elseif e.phase == :after
            delete!(pending, key)
        end
    end
    for (_, idx_pending) in pending
        push!(out[idx_pending].flags, :no_after)
    end

    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    Diagnostic(severity, title, detail, jump_to)

A single anomaly card, severity is one of `:red`, `:yellow`, `:green`.
`jump_to` is the global event index the user should inspect (or `nothing`).
"""
struct Diagnostic
    severity :: Symbol
    title    :: String
    detail   :: String
    jump_to  :: Union{Int, Nothing}
end

"""
    run_diagnostics(events, results) -> Vector{Diagnostic}

Scan an enriched-event stream once and emit one card per detected anomaly
class. Stops after the first match of each kind to avoid drowning the user
in duplicates. Always emits a green "all clear" card if nothing fires.

Extending: push your own `Diagnostic(...)` from a new pattern. The function
is plain Julia, no macros.
"""
function run_diagnostics(events, results)
    diags = Diagnostic[]

    for e in events
        if :nan in e.flags || :inf in e.flags
            push!(
                diags,
                Diagnostic(
                    :red,
                    "Non-finite value detected",
                    "Event #$(e.idx) ($(e.kind)/$(e.phase)) on $(something(e.node_id, e.variable_id, "?")) produced $(:nan in e.flags ? "NaN" : "Inf"). This usually means a numerically unstable rule (e.g. negative precision, log of zero).",
                    e.idx,
                ),
            )
            break
        end
    end

    for e in events
        if :variance_explode in e.flags
            push!(
                diags,
                Diagnostic(
                    :yellow,
                    "Variance is exploding",
                    "Event #$(e.idx) on $(something(e.node_id, e.variable_id, "?")) shows variance > 10× the previous step on the same edge. This often signals diverging beliefs or a poor initialization.",
                    e.idx,
                ),
            )
            break
        end
    end

    for e in events
        if :vague_late in e.flags
            push!(
                diags,
                Diagnostic(
                    :yellow,
                    "Posterior remained vague after iteration 1",
                    "$(e.variable_id) is still vague at iter $(e.iteration). A rule may be missing or the variable is unidentifiable from the data.",
                    e.idx,
                ),
            )
            break
        end
    end

    for e in events
        if :no_after in e.flags
            push!(
                diags,
                Diagnostic(
                    :red,
                    "Unmatched :before event (rule errored)",
                    "Event #$(e.idx) ($(e.kind)) fired :before but no matching :after was recorded. The rule likely threw mid-execution.",
                    e.idx,
                ),
            )
            break
        end
    end

    fe = nothing
    try
        fe = results.free_energy
    catch
    end
    if fe !== nothing && length(fe) > 1
        increases = findall(i -> fe[i] > fe[i - 1] + 1e-9, 2:length(fe))
        if !isempty(increases)
            push!(
                diags,
                Diagnostic(
                    :yellow,
                    "Free energy is non-monotonic",
                    "Free energy increased at iteration(s) $(join(increases .+ 1, ", ")). VMP free energy should monotonically decrease; this signals a numerical issue or non-conjugate update.",
                    nothing,
                ),
            )
        end
    end

    if isempty(diags)
        push!(
            diags,
            Diagnostic(
                :green,
                "No anomalies detected",
                "All $(length(events)) events look healthy. Free energy is monotone (if computed) and no NaN/Inf/vague-late patterns were found.",
                nothing,
            ),
        )
    end

    return diags
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Filtering & aggregation
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_filters(events; var, node, kind, phase) -> Vector{EnrichedEvent}

Narrow an enriched event stream. Each filter accepts the sentinel string
`"(any)"`, `"all"`, or `"both"` to mean *don't filter on this dimension*.

# Arguments
- `var`   : `"V1"`, …, or `"(any)"`
- `node`  : `"N1"`, …, or `"(any)"`
- `kind`  : `"rule" | "prod2" | "prodN" | "marginal"`, or `"all"`
- `phase` : `"before" | "after"`, or `"both"`
"""
function apply_filters(
    events; var = "(any)", node = "(any)", kind = "all", phase = "both"
)
    function pred(e)
        var == "(any)" || e.variable_id == var || return false
        node == "(any)" || e.node_id == node || return false
        kind == "all" || string(e.kind) == kind || return false
        phase == "both" || string(e.phase) == phase || return false
        return true
    end
    return filter(pred, events)
end

"""
    filter_events(trace, index::GraphIndex; var, node, kind, phase) -> Vector{EnrichedEvent}

Convenience: enrich a raw `trace` and apply filters in one call. Use this
from a script when you don't need to keep the full enriched stream around.
"""
filter_events(trace, idx::GraphIndex; kwargs...) = apply_filters(
    enrich(trace, idx); kwargs...
)

"""
    aggregate(events; collapse::Bool) -> Vector{Tuple}

Collapse repeated events by `(iteration, node_id, variable_id, kind, phase)`.
The summary fields are taken from the **last** event in each group; flags
are unioned across the group. Each row is a tuple
`(idx, iter, node_id, edge_ids, kind, phase, count, summary, flags, all_idxs)`.
"""
function aggregate(events; collapse::Bool)
    collapse || return [
        (
            e.idx,
            e.iteration,
            e.node_id,
            e.edge_ids,
            e.kind,
            e.phase,
            1,
            e.summary,
            e.flags,
            [e.idx],
        ) for e in events
    ]
    groups = Dict{Tuple, Vector{EnrichedEvent}}()
    order  = Tuple[]
    for e in events
        key = (e.iteration, e.node_id, e.variable_id, e.kind, e.phase)
        haskey(groups, key) || push!(order, key)
        push!(get!(groups, key, EnrichedEvent[]), e)
    end
    return [
        begin
            es = groups[k]
            last = es[end]
            all_flags = unique(
                reduce(vcat, (e.flags for e in es); init = Symbol[])
            )
            (
                last.idx,
                last.iteration,
                last.node_id,
                last.edge_ids,
                last.kind,
                last.phase,
                length(es),
                last.summary,
                all_flags,
                [e.idx for e in es],
            )
        end for k in order
    ]
end

"""
    flag_severity(flags::Vector{Symbol}) -> Symbol

Map a flag list to a single severity (`:red` > `:yellow` > `:green`).
Used to colour diagnostic cards and table rows.
"""
function flag_severity(flags::Vector{Symbol})
    any(f -> f in (:nan, :inf, :no_after), flags) && return :red
    any(
        f -> f in (:variance_explode, :variance_collapse, :vague_late), flags
    ) && return :yellow
    return :green
end

# Keep the underscored alias used internally elsewhere.
const _flag_severity = flag_severity

# ─────────────────────────────────────────────────────────────────────────────
# 5. Markdown report
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_report(events, diagnostics, results, index::GraphIndex) -> String

Produce a Markdown summary of an inference run: counts, free-energy
trajectory, all diagnostic cards, final marginals per random variable, and
the most recently flagged events. Designed to be pasted into a GitHub issue.
"""
function build_report(events, diagnostics, results, idx::GraphIndex)
    io = IOBuffer()
    println(io, "# RxInfer diagnostic report")
    println(io)
    println(io, "- Events recorded: **$(length(events))**")
    println(
        io,
        "- Iterations detected: **$(isempty(events) ? 0 : maximum(e.iteration for e in events))**",
    )
    println(
        io,
        "- Nodes: **$(length(idx.nodes))**, Variables: **$(length(idx.vars))**, Edges: **$(length(idx.edge_ids))**",
    )
    fe = try
        results.free_energy
    catch
        ;
        nothing
    end
    if fe !== nothing
        println(io, "- Free energy: ", join(round.(fe; sigdigits = 5), ", "))
    end
    println(io)
    println(io, "## Diagnostics")
    for d in diagnostics
        println(
            io,
            "- **[$(uppercase(string(d.severity)))]** $(d.title) — $(d.detail)",
        )
    end
    println(io)
    println(
        io, "## Final marginals (last :after_marginal_computation per variable)"
    )
    seen = Set{String}()
    for e in reverse(events)
        e.kind == :marginal && e.phase == :after || continue
        e.variable_id === nothing && continue
        e.variable_id in seen && continue
        push!(seen, e.variable_id)
        println(
            io,
            "- `$(e.variable_id)`: $(e.summary.dist_type) · mean=$(round(e.summary.mean; sigdigits=5)) · var=$(round(e.summary.var; sigdigits=5))",
        )
    end
    println(io)
    println(io, "## Recently flagged events")
    flagged = filter(e -> !isempty(e.flags), events)
    for e in last(flagged, min(5, length(flagged)))
        target = something(e.node_id, e.variable_id, "?")
        println(
            io,
            "- #",
            e.idx,
            " iter ",
            e.iteration,
            " ",
            e.kind,
            "/",
            e.phase,
            " on ",
            target,
            " — flags: ",
            join(e.flags, ","),
        )
    end
    return String(take!(io))
end
