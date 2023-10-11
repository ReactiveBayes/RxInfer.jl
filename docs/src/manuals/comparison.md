# Comparison to other packages

There are numerous probabilistic programming languages and packages available. While all are centered around Bayesian Inference, their approaches to the problem differ. This section offers a comparison of `RxInfer.jl` with other popular probabilistic programming languages and packages, aiming to assist potential users in understanding the differences and selecting the one that best aligns with their needs.

**DISCLAIMER**: 
1. This comparison is not exhaustive and reflects the author's experience with the packages. Some might be more thoroughly tested than others. If you're an author of one of these packages and feel the comparison is unjust, please reach out, and we'll gladly make adjustments.
2. The comparison leans more towards qualitative than quantitative, given the challenges of maintaining benchmarking code for continuously evolving packages.


| Toolbox                  | Universality | Speed | Modularity | Inference Engine      | Expressiveness | Debugging & Visualization | Language | Community & Ecosystem | 
|--------------------------|--------------|-------|------------|-----------------------|----------------|---------------------------|----------|-----------------------|
| **RxInfer.jl**           | –            | ✓     | ✓          | Message-passing       | ✓              | –                         | Julia    | –                     |
| **Turing.jl**            | ✓            | –     | ✓          | Sampling              | ✓              | –                         | Julia    | ✓                     |
| **Numpyro**              | ✓            | ✓     | ✓          | Variational Inference | –              | ✓                         | Python   | ✓                     |
| **PyMC**                 | ✓            | –     | ✓          | Sampling              | –              | ✓                         | Python   | ✓                     |
| **TensorFlow Probability** | ✓          | –     | ✓          | Sampling              | –              | ✓                         | Python   | ✓                     |
| **Infer.net**            | –            | ✓     | ✓          | Message-passing       | –              | ✓                         | C#       | –                     |
```

**Notes**:
- **Universality**: Ability to represent a broad spectrum of probabilistic models.
- **Speed**: Emphasis on computational efficiency. Note: The use of "–" for sampling in the context of this table implies a perceived inefficiency, though sampling methods are foundational in Bayesian inference.
- **Modularity**: Enables the construction of models by merging smaller, reusable components.
- **Inference Engines**: Describes the primary inference mechanism employed by the toolbox.
- **Expressiveness**: Evaluates the capability to articulate complex probabilistic models succinctly.
- **Debugging & Visualization**: Assesses the tools available for model debugging and visualization.
- **Language**: Specifies the programming language associated with the toolbox.
- **Community & Ecosystem**: Indicates the presence of a robust ecosystem, including tools, libraries, and community support.


## RxInfer.jl breakdown
**Universality** RxInfer.jl is great when it comes to building complex models formed by distributions from exponential families. 