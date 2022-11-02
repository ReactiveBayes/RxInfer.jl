# This file contains titles and descriptions for each example in this folder
# These meta information will be used for our documentation pipeline

# IMPORTANT: 0. IF YOU MAKE CHANGES TO THIS INSTRUCTION CHANGE THE DOCUMENTATION SECTION AS WELL (`docs/contributing/new-example.md`)
#            1. Make sure that the very first cell of the notebook contains ONLY `# <title>` in it and has markdown type
#               This is important for link generation in the documentation
#            2. Paths must be local and cannot be located in subfolders
#            3. Description is used to pre-generate an Examples page overview in the documentation
#            4. Use hidden option to not include a certain example in the documentation (build will still run to ensure the example runs)
#            5. Name `Overview` is reserved, please do not use it
#            6. Use $$\begin{aligned} <- on the same line, otherwise formulas will not render correctly in the documentation 
#                   <latex formulas here>
#                   \end{aligned}$$   <- on the same line (check other examples if you are not sure)
#            7. Notebooks and plain Julia have different scoping rules for global variables, if it happens so that examples generation fails due to 
#               `UndefVarError` and scoping issues use `let ... end` blocks to enforce local scoping (see `Gaussian Mixtures Multivariate.ipynb` as an example)
#            8. All examples must use and activate local `Project.toml` in the second cell (see `1.`), if you need some package add it to the `(examples)` project

return [
    (
        path = "Coin Toss Model.ipynb",
        title = "Coin toss model (Beta-Bernoulli)",
        description = "An example of Bayesian inference in Beta-Bernoulli model with IID observations.",
        hidden = false
    ),
    (
        path = "Linear Regression.ipynb",
        title = "Bayesian Linear Regression",
        description = "An example of Bayesian linear regression.",
        hidden = false
    ),
    (
        path = "Assessing People Skills.ipynb",
        title = "Assessing People’s Skills",
        description = "The demo is inspired by the example from Chapter 2 of Bishop's Model-Based Machine Learning book. We are going to perform an exact inference to assess the skills of a student given the results of the test.",
        hidden = false
    ),
    (
        path = "Gaussian Linear Dynamical System.ipynb",
        title = "Gaussian Linear Dynamical System",
        description = "An example of inference procedure for Gaussian Linear Dynamical System with multivariate noisy observations using Belief Propagation (Sum Product) algorithm. Reference: [Simo Sarkka, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).",
        hidden = false
    ),
    (
        path  = "Hidden Markov Model.ipynb",
        title = "Ensemble Learning of a Hidden Markov Model",
        description = "An example of structured variational Bayesian inference in Hidden Markov Model with unknown transition and observational matrices.",
        hidden = false
    ),
    (
        path  = "Autoregressive Model.ipynb", 
        title = "Autoregressive Model", 
        description = "An example of variational Bayesian Inference on full graph for Autoregressive model. Reference: [Albert Podusenko, Message Passing-Based Inference for Time-Varying Autoregressive Models](https://www.mdpi.com/1099-4300/23/6/683).",
        hidden = false
    ),
    (
        path = "Hierarchical Gaussian Filter.ipynb",
        title = "Hierarchical Gaussian Filter",
        description = "An example of online inference procedure for Hierarchical Gaussian Filter with univariate noisy observations using Variational Message Passing algorithm. Reference: [Ismail Senoz, Online Message Passing-based Inference in the Hierarchical Gaussian Filter](https://ieeexplore.ieee.org/document/9173980).",
        hidden = false
    ),
    (
        path = "Infinite Data Stream.ipynb",
        title = "Infinite Data Stream",
        description = "This example shows RxInfer capabilities of running inference for infinite time-series data.",
        hidden = false
    ),
    (
        path = "Identification Problem.ipynb",
        title = "System Identification Problem",
        description = "This example attempts to identify and separate two combined signals.",
        hidden = false
    ),
    (
        path = "Gaussian Mixture Univariate.ipynb",
        title = "Univariate Gaussian Mixture Model",
        description = "This example implements variational Bayesian inference in a univariate Gaussian mixture model with mean-field assumption.",
        hidden = false
    ),
    (
        path = "Gaussian Mixtures Multivariate.ipynb",
        title = "Multivariate Gaussian Mixture Model",
        description = "This example implements variational Bayesian inference in a multivariate Gaussian mixture model with mean-field assumption.",
        hidden = false,
    ),
    (
        path = "Gamma Mixture.ipynb",
        title = "Gamma Mixture Model",
        description = "This example implements one of the Gamma mixture experiments outlined in https://biaslab.github.io/publication/mp-based-inference-in-gmm/ .",
        hidden = false
    ),
    (
        path = "Global Parameter Optimisation.ipynb",
        title = "Global Parameter Optimisation",
        description = "This example shows how to use RxInfer.jl automated inference within other optimisation packages such as Optim.jl.",
        hidden = false
    ),
    (
        path = "Invertible Neural Network Tutorial.ipynb",
        title = "Invertible neural networks: a tutorial",
        description = "An example of variational Bayesian Inference with invertible neural networks. Reference: Bart van Erp, Hybrid Inference with Invertible Neural Networks in Factor Graphs.",
        hidden = false  
    ),
    (
        path = "Conjugate-NonConjugate Variational Message Passing.ipynb",
        title = "Conjugate-NonConjugate Variational Message Passing (CVI)",
        description = "This example provides an extensive tutorial for the non-conjugate message-passing based inference by exploiting the local CVI approximation.",
        hidden = false
    ),
    (
        path = "GPRegression by SSM.ipynb",
        title = "Solve GP regression by SDE",
        description = "In this notebook, we solve a GP regression problem by using 'Stochastic Differential Equation' (SDE). This method is well described in the dissertation 'Stochastic differential equation methods for spatio-temporal Gaussian process regression.' by Arno Solin and 'Sequential Inference for Latent Temporal Gaussian Process Models' by Jouni Hartikainen.",
        hidden = false  
    ),
    (
        path = "Nonlinear Noisy Pendulum.ipynb",
        title = "Nonlinear Smoothing: Noisy Pendulum",
        description = "In this demo, we will look at a realistic dynamical system with nonlinear state transitions: tracking a noisy single pendulum. We translate a differential equation in state-space model form to a probabilistic model.",
        hidden = false
    ),
    (
        path = "Nonlinear Rabbit Population.ipynb",
        title = "Nonlinear Smoothing: Rabbit Population",
        description = "In this demo, we will look at dynamical systems with nonlinear state transitions. We will start with a one-dimensional problem; the number of rabbits on an island. This problem seems overly simple, but it is a good way to demonstrate the basic pipeline of working with RxInfer.",
        hidden = false
    ),
    (
        path = "Nonlinear Virus Spread.ipynb",
        title = "Nonlinear Virus Spread",
        description = "In this demo we consider a model for the spead of a virus (not COVID-19!) in a population. We are interested in estimating the reproduction rate from daily observations of the number of infected individuals.",
        hidden = false
    ),
    (
        path = "Nonlinear Sensor Fusion.ipynb",
        title = "Nonlinear Sensor Fusion",
        description = "Nonlinear object position identification using a sparse set of sensors",
        hidden = false
    ),
    (
        path = "Kalman filter with LSTM network driven dynamic.ipynb",
        title = "Kalman filter with LSTM network driven dynamic",
        description = "In this demo, we are interested in Bayesian state estimation in Nonlinear State-Space Model using the LSTM.",
        hidden = false
    ),
    (
        path = "Handling Missing Data.ipynb",
        title = "Handling Missing Data",
        description = "This example shows how to extend the set of builtin rules to support `missing` observations.",
        hidden = false
    ),
    (
        path = "Custom nonlinear node.ipynb",
        title = "Custom Nonlinear Node",
        description = "In this example we create a non-conjugate model and use a nonlinear link function between variables. We show how to extend the functionality of `RxInfer` and to create a custom factor node with arbitrary message passing update rules.",
        hidden = false
    ),
    (
        path = "Probit model (EP).ipynb",
        title = "Probit model (EP)",
        description = "In this demo we illustrate EP in the context of state-estimation in a linear state-space model that combines a Gaussian state-evolution model with a discrete observation model.",
        hidden = false
    ),
    (
        path = "RTS vs BIFM Smoothing.ipynb",
        title = "RTS vs BIFM Smoothing",
        description = "This example performs BIFM Kalman smoother on a factor graph using message passing and compares it with the RTS implementation.",
        hidden = false
    ),
    (
        path = "Advanced Tutorial.ipynb",
        title = "Advanced Tutorial",
        description = "This notebook covers the fundamentals and advanced usage of the `RxInfer.jl` package.",
        hidden = false
    ),
    (
        path  = "Tiny Benchmark.ipynb", 
        title = "Tiny Benchmark", 
        description = "Tiny Benchmark for Internal testing.",
        hidden = true
    )
]