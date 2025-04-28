# Performance Tips

Let's face it: using RxInfer to make a model can be a challenge when you're just starting, even without having to worry about runtime. But when we've build a model, we would like it to be fast. This page is here to help you do that with a running overview of performance tips and tricks.

First of, a bad analogy: if RxInfer is the vessel taking you where you want to go, Julia is the ocean we're sailing. It is a good idea to have a basic understanding of this sea's many moods, so we'll start with an overview of the most essential Julia performance tips. For a deeper dive, see [Julia’s Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-tips) .



---

## General Julia Performance Tips

**Summary:**
- avoid global variables
- write functions
- ensure functions are type-stable
- focus on steady-state performance

### Avoid Untyped Global Variables
**Why?**  
In Julia, variables that are global (i.e., not inside a function or local scope) are **dynamically typed and slow**.  
The compiler can't specialize code for unknown types, so it falls back to slower, general-purpose execution.  
In RxInfer, this can seriously slow down:
- **Model specifications** (building factor graphs)
- **Non-linear function evaluations** (where fast numerical computation is essential)

**Solution:**  
Wrap your model-building code inside functions and ensure non-linear functions are also defined in a scoped, type-stable way. Making sure functions always return the same type of variable can boost performance.

### Be Aware of Compilation Latency

**Why?**  
Julia uses **Just-In-Time (JIT)** compilation.  
The **first time** you run a model (or even parts of it like `infer!`), Julia compiles the specialized machine code.  
This can cause noticeable delays **only once**. Afterward, execution becomes extremely fast.

**Tip:**  
Don’t worry about long first-run times during development — focus on steady-state performance.

---

## Non-Linear Nodes

### Prefer Linearization Over Unscented or CUI Methods

**Why?**  
- **Linearization** approximates non-linear functions by using **local linear models** around the current estimate.
- **Unscented Transform** and **CUI (Central Unscented Integration)** are **more accurate** but involve **more function evaluations** and **matrix operations**, which are computationally heavier.

Thus, **Linearization is much faster** and usually good enough unless you need extremely high precision.

**Further Reading:**  
See the “Deterministic Nodes” section of the RxInfer documentation for more on non-linear nodes.

---

## Free Energy Computation

### Use `free_energy = Float64` Instead of `true`

**Why?**  
When you use `free_energy = true`, RxInfer needs to store more detailed internal states for diagnostic purposes.  
When you specify `free_energy = Float64`, it reduces tracking to **minimal scalar computations**, making the process **leaner and faster**.

**Tip:**  
If you don’t need full diagnostics for debugging, prefer `Float64` mode.

---

## Special Node Considerations

Choosing the right node types helps both in memory use and CPU efficiency:

### Gaussian Nodes

- `MvNormalMeanScalePrecision` is **faster** than `MvNormalMeanPrecision`
- **Why?**  
  - `MvNormalMeanScalePrecision` exploits extra structure (scale and precision separation) to avoid heavy matrix operations.
  - It results in cheaper updates and storage.

### SoftDot Node

**Why?**  
Efficient for representing weighted sums with uncertainty — useful in regression-like structures.

### Probit Node

**Why?**  
Optimized for **binary classification** with a smooth, probabilistic interpretation — avoids having to approximate logistic sigmoid unnecessarily.

### HGF (Hierarchical Gaussian Filter)

**Why?**  
Designed to model hierarchies in beliefs with **pre-optimized update rules**, saving work compared to building full hierarchical models manually.

---

## Configuration Options: `limit_stack_depth`

### Problems with `limit_stack_depth`

**Why?**  
The `limit_stack_depth` option forces RxInfer to cap how deep recursive message passing can go.  
While useful to **avoid infinite loops** in complex graphs, setting it too low:
- **Cuts off legitimate computations**
- **Forces premature message updates**
- **May break inference stability**

Use it carefully; it’s more for **debugging** deep models than regular optimization.

---

## Converting Models: Smoothing → Filtering

### How to Convert

- **Smoothing models** use future observations to improve past estimates (i.e., two-pass inference).
- **Filtering models** only use past and present data, updating beliefs in a **causal** fashion.

**To convert:**
- Remove or disable factors that depend on future observations.
- Restrict message passing to only move forward in time.

**Why?**  
Filtering is computationally lighter and better suited for real-time or online inference tasks.

---

# Summary

In **RxInfer.jl**, fast models arise from:
- Good Julia habits (type stability, no globals)
- Smart node choices (exploiting structural optimizations)
- Awareness of compilation and configuration traps

Following these practices will lead to faster, more scalable, and more reliable inference workflows.