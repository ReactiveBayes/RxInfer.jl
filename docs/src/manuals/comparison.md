# Comparison to other packages (@id comparison)

There's a plethora of probabilistic programming languages and packages available today. While all pivot around Bayesian Inference, their methodologies vary. This section juxtaposes `RxInfer.jl` against other renowned probabilistic programming languages and packages. The goal is to enlighten potential users about the nuances and guide them in choosing the package that resonates best with their requirements.

**DISCLAIMER**: 
1. This comparison isn't exhaustive and mirrors the author's hands-on experience with the packages. Some might have undergone more rigorous testing than others. If you're an author of one of these packages and believe the comparison doesn't do justice, please reach out, and we'll be more than willing to rectify.
2. The comparison is more qualitative than quantitative, considering the intricacies of upkeeping benchmarking code for perpetually evolving packages.



| Toolbox                                                              | Universality | Efficiency | Expressiveness | Debugging & Visualization | Modularity | Inference Engine | Language | Community & Ecosystem |
| -------------------------------------------------------------------- | ------------ | ---------- | -------------- | ------------------------- | ---------- | ---------------- | -------- | --------------------- |
| [**RxInfer.jl**](https://rxinfer.ml/)                                | –            | ✓          | ✓              | –                         | –          | Message-passing  | Julia    | –                     |
| [**ForneyLab.jl**](https://github.com/biaslab/ForneyLab.jl)          | –            | –          | –              | ✓                         | –          | Message-passing  | Julia    | –                     |
| [**Infer.net**](https://dotnet.github.io/infer/)                     | –            | ✓          | –              | ✓                         | –          | Message-passing  | C#       | –                     |
| [**Turing.jl**](https://turing.ml/)                                  | ✓            | –          | ✓              | –                         | –          | Sampling         | Julia    | ✓                     |
| [**PyMC**](https://www.pymc.io/welcome.html)                         | ✓            | –          | ✓              | ✓                         | –          | Sampling         | Python   | ✓                     |
| [**NumPyro**](https://num.pyro.ai/en/stable/)                        | ✓            | ✓          | –              | ✓                         | –          | Sampling         | Python   | ✓                     |
| [**TensorFlow Probability**](https://www.tensorflow.org/probability) | ✓            | –          | –              | ✓                         | –          | Sampling         | Python   | ✓                     |
| [**Stan**](https://mc-stan.org/)                                     | –            | –          | –              | ✓                         | –          | Sampling         | Stan     | ✓                     |
(Date of creation: 20/10/2023)

**Notes**:
- **Universality**: Denotes the capability to depict a vast array of probabilistic models.
- **Efficiency**: Highlights computational prowess. A "–" in this context suggests perceived sluggishness.
- **Expressiveness**: Assesses the ability to succinctly formulate intricate probabilistic models.
- **Debugging & Visualization**: Evaluates the suite of tools for model debugging and visualization.
- **Modularity**: Reflects the potential to craft models by amalgamating smaller models.
- **Inference Engines**: Pinpoints the primary inference strategy employed by the toolbox.
- **Language**: Identifies the programming language integral to the toolbox.
- **Community & Ecosystem**: Signifies the vibrancy of the ecosystem, inclusive of tools, libraries, and community backing.

---

# RxInfer.jl breakdown

- **Universality**: `RxInfer.jl` shines in formulating intricate models derived from the exponential family distributions. The package encompasses not only commonly used distributions such as Gaussian or Bernoulli, but also specialized stochastic nodes that epitomize prevalent probabilistic models like [Autoregressive models](@ref ARmodel), [Gamma Mixture models](@ref GammaMixturemodel), among others. Furthermore, `RxInfer.jl` adeptly manages deterministic transformations of variables from the exponential family, see [Delta node](@ref delta-node-manual). Yet, for models outside the exponential family, `RxInfer.jl` might not be the prime choice. Such models would mandate the creation of novel nodes and corresponding rules, as illustrated [in](@id create-node).
  
- **Efficiency**: `RxInfer.jl` distinguishes itself with its inference engine rooted in reactive message passing. This modus operandi is supremely efficient, facilitating real-time propagation of updates across the system, endorsing parallelization, interruptibility, and more. However, the current rendition of `RxInfer.jl` hasn't harnessed all these potentials.

- **Modularity**: Broadly, the toolboxes in the table aren't modular in the truest sense. They don't champion the fusion of models by integrating smaller models. While `RxInfer.jl` currently doesn't support this, a solution is on the horizon:
  
```julia
@model function inner_inner(τ, y)
    y ~ Normal(τ[1], τ[2])
end

@model function inner(θ, α)
    β ~ Normal(0, 1)
    α ~ Gamma(β, 1)
    α ~ inner_inner(τ = θ)
end

@model function outer()
    local w
    for i = 1:5
        w[i] ~ inner(θ = Gamma(1, 1))
    end
    y ~ inner(θ = w[2:3])
end
```

- **Expressiveness**: `RxInfer.jl` empowers users to elegantly and succinctly craft models, closely mirroring probabilistic notation, thanks to Julia's macro capabilities. To illustrate this, let's consider the following model:

$$\begin{aligned}
 x & \sim \mathrm{Normal}(0.0, 1.0)\\
 w & \sim \mathrm{InverseGamma}(1.0, 1.0)\\
 y & \sim \mathrm{Normal}(x, w)
\end{aligned}$$

The model then is expressed in `RxInfer.jl` as follows:
```@example comparisondoc
using RxInfer

@model function example_model()
    x ~ Normal(mean=0.0, var=1.0)
    w ~ InverseGamma(α = 1, θ = 1)
    y ~ Normal(mean = x, var = w)
end
```

- **Debugging & Visualization**: While `RxInfer.jl` grapples with Julia's nascent debugging system, it does proffer a mechanism to debug the inference [procedure](@ref user-guide-debugging), albeit not as seamlessly as some other packages.