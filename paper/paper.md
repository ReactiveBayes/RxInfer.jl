---
title: 'RxInfer: A Julia package for reactive real-time Bayesian inference'
tags:
  - Julia
  - statistics
  - Bayesian inference
  - variational optimization
  - message passing
authors:
  - name: Dmitry Bagaev
    orcid: 0000-0000-0000-0000
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1  
  - name: Albert Podusenko
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Bert de Vries
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Technical University of Eindhoven
   index: 1
date: 9 December 2022
bibliography: paper.bib

# Citations to entries in paper.bib should be in
# [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
# format.
# 
# If you want to cite a software repository URL (e.g. something on GitHub without a preferred
# citation) then you can do it with the example BibTeX entry below for @fidgit.
# 
# For a quick reference, the following citation commands can be used:
# - `@author:2001`  ->  "Author et al. (2001)"
# - `[@author:2001]` -> "(Author et al., 2001)"
# - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures can be included like this:
# ![Caption for example figure.\label{fig:example}](figure.png)
# and referenced from text using \autoref{fig:example}.
# 
# Figure sizes can be customized by adding an optional second parameter:
# ![Caption for example figure.](figure.png){ width=20% }
---

# Summary

Bayesian inference realizes optimal information processing through a full commitment to reasoning by probability theory. 
The Bayesian framework is positioned at the core of modern AI technology for applications such as speech and image recognition and generation, medical analysis, robot navigation, etc.,
as it describes how a rational agent ought to update beliefs when new information is revealed by the agent's environment. 
Unfortunately, perfect Bayesian reasoning is generally intractable, since it requires the calculation of (often) very high-dimensional integrals. 
As a result, a number of numerical algorithms for approximating Bayesian inference have been developed and implemented in probabilistic programming packages. 
Successful methods include variants of Monte Carlo (MC) sampling, Variational Inference (VI), and Laplace approximation. 

We present **RxInfer.jl**, which is a Julia [] package for real-time variational Bayesian
  inference based on reactive message passing in a factor graph representation of the model under
  study [].
**RxInfer.jl** provides access to a powerful model specification language that translates a textual description of a probabilistic model to a corresponding factor graph representation.
In addition, **RxInfer.jl** supports hybrid variational inference processes, where different
  Bayesian inference methods can be combined together in different parts of the model, resulting in a
  straightforward mechanism to trade off accuracy for computational speed.
The underlying implementation relies on a reactive programming paradigm and supports the processing
  of infinite asynchronous streams of data by design.
In this framework, the inference engine *reacts* to new data by automatically updating
  relevant posteriors.

Over the past few years, the inference methods in this package have been tested on many advanced
  probabilistic models, resulting in several publications in high-ranked journals such as Entropy
	  [1], Frontiers [2], and conferences like MLSP-2021 [3], ISIT-2021 [4], PGM [5], EUSIPCO [6] and
  SiPS [7].

# Statement of need

Many important AI applications, such as self-driving vehicles, weather forecasting, extended
  reality video processing, and others require continually solving an inference task in sophisticated
  probabilistic models with a large number of latent variables.
Often, the inference task in these applications must be performed continually and in real time in
  response to new observations.
Popular MC-based inference methods, such as No U-Turn Samples (NUTS) or Hamltonian Monte Carlo (HMC), 
  rely on computationally heavy sampling procedures that do not scale well to probabilistic
  models with thousands of latent states.
Therefore, MC-based inference is practically not suitable for real-time applications.
While the alternative variational inference (VI) method promises to scale better to large models
  than sampling-based inference, VI requires the derivation of gradients of the "variational Free
  Energy" cost function.
For large models, manual derivation of these gradients might be not feasible, while automated
  "black-box" gradient methods do not scale either because they are not capable of taking advantage
  of sparsity or conjugate pairs in the model.
Therefore, while Bayesian inference is known as the optimal data processing framework, in practice,
  real-time AI applications rely on much simpler, often ad hoc, data processing algorithms.

# Solution proposal and overview of functionality

We present **RxInfer.jl**, a package for processing infinite data streams by real-time
  Bayesian inference in large probabilistic models.
**RxInfer.jl** implements variational Bayesian inference as a variational Constrained Bethe Free Energy (CBFE) functional optimization process [].
The underlying inference engine derives its speed from taking advantage of both statistical
  independencies and conjugate pairings of variables in the factor graph.
Through an automated reactive message passing in the graph, where each message carves away a bit of
  the variational Free Energy cost function. When derivations for specific nodes are available the probabilistic models can be
  combined hierarchically without need to derive optimization procedure from scratch.

**RxInfer.jl** is an open source package, available at [https://github.com/biaslab/RxInfer.jl](https://github.com/biaslab/RxInfer.jl), and enjoys the following features:


- A user-friendly probabilistic model specification.
      Through Julia macros, **RxInfer.jl** is capable of automatically transforming a textual
        description of a probabilistic model to a factor graph representation of that model.
- A hybrid inference engine.
      The inference engine supports a variety of well-known message passing-based inference methods such
        as belief propagation, structured and mean-field variational message passing, expectation
        propagation, expectation maximization, and conjugate-computation variational inference (CVI).
- A customized trade-off between accuracy and speed.
      For each location (node and edge) in the graph, **RxInfer.jl** enables a custom specification
        of inference constraints on the variational family of distributions in the CBFE optimization
        procedure.
      This enables the use of different Bayesian inference methods at different locations of the graph,
        leading to an optimized trade-off between accuracy and speed.
- Support for real-time processing of infinite data streams.
      **RxInfer.jl** is based on the reactive programming paradigm that enables asynchronous data processing
      as soon as data arrives.
- Support for large static data sets.
      The package is not limited to real-time processing of data streams and also scales well to batch
        processing of large data sets and large probabilistic models that can include hundreds of thousands
        of latent variables [].
- **RxInfer.jl** is extensible.
      The public API defines a user-friendly and straightforward way to extend the built-in functionality
        with custom nodes and message update rules.
- A large collection of pre-computed analytical inference solutions.
      Current built-in functionality includes fast inference solutions for linear Gaussian dynamical
        systems, auto-regressive models, hierarchical models, discrete-valued models, mixture models,
        invertible neural networks, arbitrary non-linear state transition functions, and conjugate pair
        primitives.
- The resulting inference procedure is auto-differentiable with external packages, such as
      **ForwardDiff.jl** or **ReverseDiff.jl**
- The inference engine supports different floating point number types, such as `Float32`,
      `Float64`, and `BigFloat`

A large collection of examples is available at
  [https://biaslab.github.io/RxInfer.jl/stable/examples/overview/](https://biaslab.github.io/RxInfer.jl/stable/examples/overview/).

# Example usage


In this section we show a small example usage based on the Example 3.7 in Sarkka [], where the goal
  is to estimate the state (angle) of a simple pendulum system in real-time.
The differential equations and state transition of this model are non-linear and have no closed
  form analytical solution.
The differential equations for a simple pendulum look like:
  \begin{equation}
	  \frac{\mathrm{d}^2\alpha}{\mathrm{d}t^2} = -g \sin(\alpha) + \omega(t),\label{eq:pendulum_diff}
  \end{equation}
  where $\alpha$ is the angle of the pendulum, $t$ is the time, $g$ is the
  gravitational acceleration, $\omega(t)$ is a random noise process.
The equation \autoref{eq:pendulum_diff} seen to be a special case of continuous-time non-linear dynamic
  model of the form
  \begin{equation}
	  \frac{\mathrm{d}x}{\mathrm{d}t} = f(x) + Lw,
  \end{equation}
  where $x$ is a two-dimensional vector $\begin{bmatrix} x^{(1)} \\ x^{(2)}\end{bmatrix}\equiv\begin{bmatrix}\alpha \\ v\end{bmatrix}$ 
  with $\alpha$ and $v$ being the angle and velocity respectively.
We use a simple forward finite-difference approximation scheme $\mathrm{d}x / \mathrm{d}t = (x_{t+1} - x_{t}) / \Delta t$ to write a probabilistic model of the non-linear pendulum in the
  following form:
  \begin{equation}
	  \begin{bmatrix}
		  \alpha_{t + 1} \\ v_{t + 1}
	  \end{bmatrix}
	  =
	  \begin{bmatrix}
		  \alpha_{t} + v_{t} \Delta t \\ v_t - g \sin(\alpha_t) \Delta t
	  \end{bmatrix}
	  ,
  \end{equation}
  hence
  \begin{equation}
	  f(x) =
	  \begin{bmatrix}
		  x^{(1)} + x^{(2)} \Delta t \\ x^{(2)}
		  - g \cdot \sin(x^{(1)}) \Delta t
	  \end{bmatrix}.
  \end{equation}
  

We use the `@model` macro to write down the corresponding probabilistic model, the
  `@meta` macro to specify an approximation method for the non-linearity in the model, the
  `@constraints` macro to write down constraints for the variational family of distributions
  for the Bethe Free Energy optimization procedure, and the `@autoupdates` macro to specify
  how to update priors about the current state of the system.
Finally, we use the `inference` function to execute the inference. The inference procedure runs in real-time and
takes 166 microseconds per observation on a regular office laptop (166 milliseconds in total for 1000 observations).

```julia
f(x) = [x[1] + x[2] * Δt, x[2] - G * sin(x[1]) * Δt]

# We use the `@model` macro to define the probabilistic model 
@model function pendulum()
    # Define reactive inputs for the `prior` 
    # of the current angle state
    prior_mean = datavar(Vector{Float64})
    prior_cov  = datavar(Matrix{Float64})

    previous_state ~ MvNormal(μ = prior_mean, Σ = prior_cov)
    # Use `f` as state transition function
    state ~ g(previous_state)

    # Assign a prior for the noise component
    noise_shape = datavar(Float64)
    noise_scale = datavar(Float64)
    noise ~ Gamma(noise_shape, noise_scale)

    # Define reactive input for the `observation`
    observation = datavar(Float64)
    observation ~ Normal(μ = dot([1.0, 0.0], state), τ = noise)
end
```

```julia
@constraints function pendulum_constraint()
    # Assume the `state` and the `noise` are independent
    q(state, noise) = q(state)q(noise)
end
```

```julia
@meta function pendulum_meta()
    # Use the `Linearization` approximation method 
    # around the (potentially) non-linear function `f`
    f() -> Linearization()
end
```

```julia
function experiment(observations)

    # The `@autoupdates` structure defines how to update 
    # the priors for the next observation
    autoupdates = @autoupdates begin
        # Update `prior` automatically as soon as 
        # we have a new posterior for the `state`
        prior_mean  = mean(q(state))
        prior_cov   = cov(q(state))
        noise_shape = shape(q(noise))
        noise_scale = scale(q(noise))
    end

    results = rxinference(
        model = pendulum(),
        constraints = pendulum_constraint(),
        meta = pendulum_meta(),
        autoupdates = autoupdates,
        data = (observation = observations,),
        initmarginals = (
            # We assume a relatively good prior for the very first state
            state = MvNormalMeanPrecision([0.5, 0.0], [100.0 0.0; 0.0 100.0]),
            # And we assign a vague prior for the noise component
            noise = Gamma(1.0, 100.0)
        ),
        # We indicate that we want to keep a history of estimated 
        # states and the noise component
        historyvars = (state = KeepLast(), noise = KeepLast()),
        keephistory = length(observations),
        # We perform 5 VMP iterations on each observation
        iterations = 5,
        # We start the inference procedure automatically
        autostart = true
    )

    return results
end
```

![
The inference results for the pendulum example. X-axis represents time $t$ (in seconds). Y-axis represents the current angle of the pendulum (in radians) at time $t$. Real (un-observed)
signal is shown in blue line. Observations are shown as orange dots. The inference results are shown as green line with area, 
which represents posterior uncertainty (one standard deviation).
The inference procedure runs in real-time and takes 166 microseconds per observation on a regular office laptop (166 milliseconds in total for 1000 observations).\label{fig:example}
](inference.pdf)

# Acknowledgments

The authors gratefully acknowledge contributions and support from colleagues in the BIASlab in the Department of Electrical Engineering at the University of Eindhoven of Technology.

# References
