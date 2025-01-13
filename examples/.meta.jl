# This file contains titles and descriptions for each example in this folder
# These meta information will be used for our documentation pipeline
# Please carefully read the documentation section `Contributing: new example` for more information
return (
    categories = (
        basic_examples = (title = "Basic examples", description = "Basic examples contain \"Hello World!\" of Bayesian inference in RxInfer."),
        advanced_examples = (title = "Advanced examples", description = "Advanced examples contain more complex inference problems."),
        problem_specific = (title = "Problem specific", description = "Problem specific examples contain specialized models and inference for various domains."),
        hidden_examples = (title = "", description = "")
    ),
    examples = [
        (
            filename = "Coin Toss Model.ipynb",
            title = "Coin toss model (Beta-Bernoulli)",
            description = "An example of Bayesian inference in Beta-Bernoulli model with IID observations.",
            category = :basic_examples
        ),
        (
            filename = "Bayesian Linear Regression Tutorial.ipynb",
            title = "Bayesian Linear Regression Tutorial",
            description = "An extensive tutorial on Bayesian linear regression with RxInfer with a lot of examples, including multivariate and hierarchical linear regression.",
            category = :basic_examples
        ),
        (
            filename = "Binomial Regression.ipynb",
            title = "Binomial Regression",
            description = "An example of Bayesian inference in Binomial regression with Expectation Propagation.",
            category = :basic_examples
        ),
        (
            filename = "Kalman filtering and smoothing.ipynb",
            title = "Kalman filtering and smoothing",
            description = "In this demo, we are interested in Bayesian state estimation in different types of State-Space Models, including linear, nonlinear, and cases with missing observations",
            category = :basic_examples
        ),
        (
            filename = "Predicting Bike Rental Demand.ipynb",
            title = "Predicting Bike Rental Demand",
            description = "An illustrative guide to implementing prediction mechanisms within RxInfer.jl, using bike rental demand forecasting as a contextual example.",
            category = :basic_examples
        ),
        (
            filename = "Hidden Markov Model.ipynb",
            title = "How to train your Hidden Markov Model",
            description = "An example of structured variational Bayesian inference in Hidden Markov Model with unknown transition and observational matrices.",
            category = :basic_examples
        ),
        (
            filename = "Active Inference Mountain car.ipynb",
            title = "Active Inference Mountain car",
            description = "This notebooks covers RxInfer usage in the Active Inference setting for the simple mountain car problem.",
            category = :advanced_examples
        ),
        (
            filename = "Advanced Tutorial.ipynb",
            title = "Advanced Tutorial",
            description = "This notebook covers the fundamentals and advanced usage of the `RxInfer.jl` package.",
            category = :advanced_examples
        ),
        (
            filename = "Assessing People Skills.ipynb",
            title = "Assessing People’s Skills",
            description = "The demo is inspired by the example from Chapter 2 of Bishop's Model-Based Machine Learning book. We are going to perform an exact inference to assess the skills of a student given the results of the test.",
            category = :advanced_examples
        ),
        (
            filename = "Chance Constraints.ipynb",
            title = "Chance-Constrained Active Inference",
            description = "This notebook applies reactive message passing for active inference in the context of chance-constraints.",
            category = :advanced_examples
        ),
        (
            filename = "Conjugate-Computational Variational Message Passing.ipynb",
            title = "Conjugate-Computational Variational Message Passing (CVI)",
            description = "This example provides an extensive tutorial for the non-conjugate message-passing based inference by exploiting the local CVI approximation.",
            category = :advanced_examples
        ),
        (
            filename = "Global Parameter Optimisation.ipynb",
            title = "Global Parameter Optimisation",
            description = "This example shows how to use RxInfer.jl automated inference within other optimisation packages such as Optim.jl.",
            category = :advanced_examples
        ),
        (
            filename = "GP Regression by SSM.ipynb",
            title = "Solve GP regression by SDE",
            description = "In this notebook, we solve a GP regression problem by using 'Stochastic Differential Equation' (SDE). This method is well described in the dissertation 'Stochastic differential equation methods for spatio-temporal Gaussian process regression.' by Arno Solin and 'Sequential Inference for Latent Temporal Gaussian Process Models' by Jouni Hartikainen.",
            category = :advanced_examples
        ),
        (
            filename = "Infinite Data Stream.ipynb",
            title = "Infinite Data Stream",
            description = "This example shows RxInfer capabilities of running inference for infinite time-series data.",
            category = :advanced_examples
        ),
        (
            filename = "Nonlinear Sensor Fusion.ipynb",
            title = "Nonlinear Sensor Fusion",
            description = "Nonlinear object position identification using a sparse set of sensors",
            category = :advanced_examples
        ),
        (
            filename = "Autoregressive Models.ipynb",
            title = "Autoregressive Models",
            description = "An example of Bayesian treatment of latent AR and ARMA models. Reference: [Albert Podusenko, Message Passing-Based Inference for Time-Varying Autoregressive Models](https://www.mdpi.com/1099-4300/23/6/683).",
            category = :problem_specific
        ),
        (
            filename = "Gamma Mixture.ipynb",
            title = "Gamma Mixture Model",
            description = "This example implements one of the Gamma mixture experiments outlined in https://biaslab.github.io/publication/mp-based-inference-in-gmm/ .",
            category = :problem_specific
        ),
        (
            filename = "Gaussian Mixture.ipynb",
            title = "Gaussian Mixture",
            description = "This example implements variational Bayesian inference in univariate and multivariate Gaussian mixture models with mean-field assumption.",
            category = :problem_specific
        ),
        (
            filename = "Hierarchical Gaussian Filter.ipynb",
            title = "Hierarchical Gaussian Filter",
            description = "An example of online inference procedure for Hierarchical Gaussian Filter with univariate noisy observations using Variational Message Passing algorithm. Reference: [Ismail Senoz, Online Message Passing-based Inference in the Hierarchical Gaussian Filter](https://ieeexplore.ieee.org/document/9173980).",
            category = :problem_specific
        ),
        (
            filename = "Invertible Neural Network Tutorial.ipynb",
            title = "Invertible neural networks: a tutorial",
            description = "An example of variational Bayesian Inference with invertible neural networks. Reference: Bart van Erp, Hybrid Inference with Invertible Neural Networks in Factor Graphs.",
            category = :problem_specific
        ),
        (
            filename = "Probit Model (EP).ipynb",
            title = "Probit Model (EP)",
            description = "In this demo we illustrate EP in the context of state-estimation in a linear state-space model that combines a Gaussian state-evolution model with a discrete observation model.",
            category = :problem_specific
        ),
        (
            filename = "RTS vs BIFM Smoothing.ipynb",
            title = "RTS vs BIFM Smoothing",
            description = "This example performs BIFM Kalman smoother on a factor graph using message passing and compares it with the RTS implementation.",
            category = :problem_specific
        ),
        (
            filename = "Simple Nonlinear Node.ipynb",
            title = "Simple Nonlinear Node",
            description = "In this example we create a non-conjugate model and use a nonlinear link function between variables. We show how to extend the functionality of `RxInfer` and to create a custom factor node with arbitrary message passing update rules.",
            category = :problem_specific
        ),
        (
            filename  = "Universal Mixtures.ipynb", 
            title = "Universal Mixtures", 
            description = "Universal mixture modeling.",
            category = :problem_specific
        ),
        (
            filename  = "Litter Model.ipynb", 
            title = "Litter Model", 
            description = "Using Bayesian Inference and RxInfer to estimate daily litter events (adapted from https://learnableloop.com/posts/LitterModel_PORT.html)",
            category = :problem_specific
        ),
        (
            filename  = "Structural Dynamics with Augmented Kalman Filter.ipynb", 
            title = "Structural Dynamics with Augmented Kalman Filter", 
            description = "In this example, we estimate system states and unknown input forces for a simple **structural dynamical system** using the Augmented Kalman Filter (AKF) (https://www.sciencedirect.com/science/article/abs/pii/S0888327011003931) in **RxInfer**.",
            category = :problem_specific
        ),
        (
            filename  = "Tiny Benchmark.ipynb", 
            title = "Tiny Benchmark", 
            description = "Tiny Benchmark for Internal testing.",
            category = :hidden_examples
        ),
    ]
)
