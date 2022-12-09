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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Technical University of Eindhoven
   index: 1
 - name: TODO 
   index: 2
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

# Problem statement

Many important AI applications, such as self-driving vehicles and extended reality video processing, require real-time Bayesian inference. 
However, sampling-based inference methods do not scale well to realistic probabilistic models with a significant number of latent states. 
As a result, Monte Carlo sampling-based methods are not suitable for real-time applications. 
Variational Inference promises to scale better than sampling-based inference, but VI requires derivation of gradients of a "variational Free Energy" cost function. 
For large models, manual derivation of these gradients is not feasible, and automated "black-box" gradient methods are too inefficient to be applicable to real-time inference applications. 
Therefore, while Bayesian inference is known as the optimal data processing framework, in practice, real-time AI applications rely on much simpler, often ad hoc, data processing algorithms. 

# Solution proposal

We present RxInfer.jl, a package for processing infinite data streams by real-time Bayesian inference in large probabilistic models. RxInfer is open source, available at http://rxinfer.ml, and enjoys the following features:

- A flexible probabilistic model specification. Through Julia macros, RxInfer is capable of automatically transforming a textual description of a probabilistic model to a factor graph representation of that model.
- A flexible inference engine. The inference engine supports a variety of well-known message passing-based inference methods such as belief propagation, structured and mean-field variational message passing, expectation propagation, etc.
- A customized trade-off between accuracy and speed. For each (node and edge) location in the graph, RxInfer enables a custom specification of inference constraints on the variational family of distributions. This enables the use of different Bayesian inference methods at different locations of the graph, leading to an optimized trade-off between accuracy and computational complexity.
- Support for real-time processing of infinite data streams. Since RxInfer is based on a reactive programming framework, implemented by the package Rocket.jl, an ongoing inference process is always interruptible and an inference result is always available.
- Support for large static data sets. The package is not limited to real-time processing of data streams and also scales well to batch processing of large data sets.
- RxInfer is extensible. A large and extendable collection of precomputed analytical inference solutions for standard problems increases the efficiency of the inference process. Current methods include solutions for linear Gaussian dynamical systems, auto-regressive Models, Gaussian and Gamma mixture models, convolution of distributions, and conjugate pair primitives.

# Evaluation 

Over the past few years, the ecosystem has been tested on many advanced probabilistic models that have led to several publications in high-ranked journals such as Entropy [1], Frontiers [2],
and conferences like MLSP-2021 [3], ISIT-2021 [4], PGM[5], EUSIPCO [6] and SiPS[7]. 

# Acknowledgements

# References
