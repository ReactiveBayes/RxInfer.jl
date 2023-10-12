# Comparison to other packages

There are numerous probabilistic programming languages and packages available. While all are centered around Bayesian Inference, their approaches to the problem differ. This section offers a comparison of `RxInfer.jl` with other popular probabilistic programming languages and packages, aiming to assist potential users in understanding the differences and selecting the one that best aligns with their needs.

**DISCLAIMER**: 
1. This comparison is not exhaustive and reflects the author's experience with the packages. Some might be more thoroughly tested than others. If you're an author of one of these packages and feel the comparison is unjust, please reach out, and we'll gladly make adjustments.
2. The comparison leans more towards qualitative than quantitative, given the challenges of maintaining benchmarking code for continuously evolving packages.

| Toolbox                  | Universality | Speed | Expressiveness | Debugging & Visualization | Modularity | Inference Engine      | Language | Community & Ecosystem | 
|--------------------------|--------------|-------|----------------|---------------------------|------------|-----------------------|----------|-----------------------|
| **RxInfer.jl**           | –            | ✓     | ✓              | –                         | –          | Message-passing       | Julia    | –                     |
| **Turing.jl**            | ✓            | –     | ✓              | –                         | –          | Sampling              | Julia    | ✓                     |
| **PyMC**                 | ✓            | –     | –              | ✓                         | –          | Sampling              | Python   | ✓                     |
| **Numpyro**              | ✓            | ✓     | –              | ✓                         | –          | Variational Inference | Python   | ✓                     |
| **TensorFlow Probability** | ✓          | –     | –              | ✓                         | –          | Variational Inference              | Python   | ✓                     |
| **Infer.net**            | –            | ✓     | –              | ✓                         | –          | Message-passing       | C#       | –                     |


**Notes**:
- **Universality**: Ability to represent a broad spectrum of probabilistic models.
- **Speed**: Emphasis on computational efficiency. Note: The use of "–" for sampling in the context of this table implies a perceived inefficiency.
- **Expressiveness**: Evaluates the capability to articulate complex probabilistic models succinctly.
- **Debugging & Visualization**: Assesses the tools available for model debugging and visualization.
- **Modularity**: Enables the construction of models by merging smaller, reusable components.
- **Inference Engines**: Describes the primary inference mechanism employed by the toolbox.
- **Language**: Specifies the programming language associated with the toolbox.
- **Community & Ecosystem**: Indicates the presence of a robust ecosystem, including tools, libraries, and community support.

---

# RxInfer.jl breakdown

- **Universality**: `RxInfer.jl` excels in building complex models formed of distributions from the exponential family. The package also implements various specialized stochastic nodes representing common probabilistic models such as Autoregressive models, Gamma Mixture models, and more. `RxInfer.jl` handles deterministic transformations of variables from the exponential family well, [see](@ref delta-node-manual). However, for models not formed by distributions from the exponential family, `RxInfer.jl` might not be the optimal choice. Building such models would necessitate creating new nodes and corresponding rules as [in](@id create-node).
  
- **Speed**: `RxInfer.jl` stands out in terms of speed. 

- **Modularity**: `RxInfer.jl` is modular, allowing users to construct models by combining smaller, reusable components. This is especially beneficial when creating complex models. (More details coming soon...)

- **Expressiveness**: With `RxInfer.jl`, users can craft intricate models in an elegant and concise manner thanks to macros and the Julia language. 

- **Debugging & Visualization**: `RxInfer.jl` provides a way to debug the inference [procedure](@ref user-guide-debugging). Though not as convinient as in other packages.

- **Community & Ecosystem**: Being a relatively new package, `RxInfer.jl` is still in its infancy. However, it's built atop the burgeoning Julia ecosystem. (More details coming soon...)