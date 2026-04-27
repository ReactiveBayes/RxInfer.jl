# Plan: Implement `Base.show` for ReactiveMP / RxInfer events

Resolves:
- [ReactiveMP.jl#599](https://github.com/ReactiveBayes/ReactiveMP.jl/issues/599)
- [RxInfer.jl#638](https://github.com/ReactiveBayes/RxInfer.jl/issues/638)

Context: TBLogger text summaries are dominated by raw struct dumps because none of the event types or their nested fields define `Base.show`. Fixing this is split between two repos.

---

## 1) Which package?

| Package | Why work is needed |
|---|---|
| **ReactiveMP.jl** (primary) | `MessageMapping`, `Message`, `MessageProductContext`, `AnnotationDict`, `FormConstraintCheckEach/Last`, and the 10 `Event` structs in `src/callbacks.jl` — none have custom `show`. |
| **RxInfer.jl** (lightweight) | The 10 events in `src/callbacks/events.jl` (`BeforeIterationEvent`, `OnMarginalUpdateEvent`, …). Also drop the now-redundant inline `_format_fields` formatting strings in `ext/TensorBoardLoggerExt/TensorBoardLoggerExt.jl` once `show` does the work. |

Order: ReactiveMP PR first (independent), then a follow-up RxInfer PR that bumps the ReactiveMP compat bound and simplifies the extension.

---

## 2) Branches

Two new branches, one per repo:

- `ReactiveMP.jl` → `feat/event-show-methods` (resolves #599)
- `RxInfer.jl` → `feat/event-show-methods` (resolves #638) — stacked on `tb_textsummary_update` (still local, not yet in PR), so the show-methods work can land alongside the Summary-tag changes without an extra rebase.

---

## 3) Which events matter

Two-axis filter: (a) emitted by `RxInferTraceCallbacks` in real inferences, (b) currently dump raw structs.

### Tier A — RxInfer events (always fire, every run, dominate the Text tab)

- `BeforeModelCreationEvent`, `AfterModelCreationEvent`
- `BeforeInferenceEvent`, `AfterInferenceEvent`
- `BeforeIterationEvent`, `AfterIterationEvent`
- `BeforeDataUpdateEvent`, `AfterDataUpdateEvent`
- `OnMarginalUpdateEvent`
- `BeforeAutostartEvent`, `AfterAutostartEvent`

### Tier B — ReactiveMP events (fire under `addons` / debug callbacks)

- `Before/AfterMessageRuleCallEvent`
- `Before/AfterProductOfTwoMessagesEvent`, `Before/AfterProductOfMessagesEvent`
- `Before/AfterFormConstraintAppliedEvent`
- `Before/AfterMarginalComputationEvent`

### Tier C — supporting types referenced by Tier B fields (the real source of struct-dump noise per the screenshot in #638)

- `ReactiveMP.MessageMapping` ← biggest offender
- `ReactiveMP.Message` (only `DeferredMessage` has show today)
- `ReactiveMP.MessageProductContext`
- `ReactiveMP.AnnotationDict`
- `FormConstraintCheckEach`, `FormConstraintCheckLast` (1-line each)

Tier C is what unlocks Tier B — fixing the events alone leaves their fields ugly.

---

## 4) Fields to emit & how to show them

**Format convention** — single-line `EventName(k1=v1, k2=v2, …)`. Matches the existing `TracedEvent(:name)` style at `src/callbacks/trace.jl:23`. Add a 2-arg `show(io, ::MIME"text/plain", …)` only where the multi-line form genuinely helps (long messages tuples).

| Event | Fields shown | Format |
|---|---|---|
| `BeforeModelCreationEvent` | `span_id` | `BeforeModelCreationEvent(span=ab12…)` (truncate UUID to 4 chars) |
| `AfterModelCreationEvent` | `model` (just type+id), `span_id` | `AfterModelCreationEvent(model=ProbabilisticModel#42, span=ab12…)` |
| `Before/AfterInferenceEvent` | `model`, `span_id` | same |
| `Before/AfterIterationEvent` | `iteration`, `stop_iteration`, `span_id` | `BeforeIterationEvent(iter=3/10, span=…)` |
| `Before/AfterDataUpdateEvent` | `data` keys (not values), `span_id` | `BeforeDataUpdateEvent(data=[:y, :x], span=…)` |
| `OnMarginalUpdateEvent` | `variable_name`, `update` (delegate to dist's show via `getdata`) | `OnMarginalUpdateEvent(var=:θ, update=NormalMeanVariance(0.5, 0.1))` |
| `Before/AfterAutostartEvent` | `engine` type, `span_id` | `BeforeAutostartEvent(engine=RxInferenceEngine, span=…)` |
| `Before/AfterMessageRuleCallEvent` | `mapping` (uses new MessageMapping show), `messages` arity, `result` (After only), `span_id` | `AfterMessageRuleCallEvent(mapping=NormalMeanVariance:out(μ,τ), nmsgs=2, result=…, span=…)` |
| `Before/AfterProductOfTwoMessagesEvent` | `variable.label`, `left`, `right`, `result`, `span_id` | `AfterProductOfTwoMessagesEvent(var=:θ, left=…, right=…, result=…, span=…)` |
| `Before/AfterProductOfMessagesEvent` | `variable.label`, `nmessages`, `result`, `span_id` | similar |
| `Before/AfterFormConstraintAppliedEvent` | `variable.label`, `strategy` short name, `distribution`, `result`, `span_id` | `AfterFormConstraintAppliedEvent(var=:θ, strategy=CheckEach, dist→result=Normal→Normal, span=…)` |
| `Before/AfterMarginalComputationEvent` | `variable.label`, `nmessages`, `result`, `span_id` | similar |

### Tier C show methods (the real win)

- `MessageMapping` → `MessageMapping(NormalMeanVariance, :out, [:μ,:τ])` (functional form, type, factornode, vtag, names)
- `Message` → `Message(NormalMeanVariance(μ=0.5, σ²=0.1))` — delegate to inner distribution's `show`
- `MessageProductContext` → `MessageProductContext(strategy=…, fold=…)` — name and one-line strategy
- `AnnotationDict` → `AnnotationDict(n=2)` when populated, `AnnotationDict()` when empty
- `FormConstraintCheckEach` → `CheckEach`, `FormConstraintCheckLast` → `CheckLast`

### Span-id helper

Every event has one. Add an internal `_show_span(io, span_id)` that prints first 4 hex chars; UUID4 makes the full form noisy and meaningless to humans but useful for grep, so keep `show(io, ::MIME"text/plain", ...)` printing the full UUID.

### Test approach

- Golden-string tests against `repr(ev)` output for each event.
- Regression test that the EventCounts / Events text-tag in TBLogger no longer contains the substring `MessageMapping{` or `Message{` (the raw-dump telltale).

---

## Cleanup follow-up in RxInfer

Once ReactiveMP exports decent `show`, the giant inline `"variable: $(ev.variable.label) | context: $(ev.context) | …"` strings at `ext/TensorBoardLoggerExt/TensorBoardLoggerExt.jl:444-530` collapse into one call:

```julia
_log_text!(ctx, "after_marginal_computation", repr(ev); step=idx)
```

That's roughly 90 LOC removed.

---

## Sequencing options

- **(a) RxInfer-only PR first** — Tier A events, immediate ~30% noise reduction, ships independently of ReactiveMP review.
- **(b) ReactiveMP PR first** — Tier B+C, the structural fix; unblocks the cleanup in RxInfer.
- **(c) Both branches in parallel** — fastest end-to-end, two reviews to manage.
