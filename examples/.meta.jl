# This file contains titles and descriptions for each example in this folder
# These meta information will be used for our documentation pipeline

# IMPORTANT: 1. Make sure that the very first cell of the notebook contains ONLY `# <title>` in it and has markdown type
#               This is important for link generation in the documentation
#            2. Paths must be local and cannot be located in subfolders
#            3. Description is used to pre-generate an Examples page overview in the documentation
#            4. Use hidden option to not include a certain example in the documentation (build will still run to ensure the example runs)
#            5. Name `Overview` is reserved, please do not use it
#            6. Use $$\begin{aligned} <- on the same line, otherwise formulas will not render correctly in the documentation 
#                   <latex formulas here>
#                   \end{aligned}$$   <- on the same line
#            7. Notebooks and plain Julia have different scoping rules for global variables, if it happens so that examples generation fails due to 
#               `UndefVarError` and scoping issues use `let ... end` blocks to enforce local scoping (see `Gaussian Mixtures Multivariate.ipynb` as an example)
#            8. All examples must use and activate local `Project.toml`, if you need some package add it to the `(examples)` project

return [
    (
        path  = "Autoregressive model.ipynb", 
        title = "Autoregressive Model", 
        description = "An example of variational Bayesian Inference on full graph for Autoregressive model. Reference: [Albert Podusenko, Message Passing-Based Inference for Time-Varying Autoregressive Models](https://www.mdpi.com/1099-4300/23/6/683).",
        hidden = false
    ),
    (
        path = "Gaussian Mixtures Multivariate.ipynb",
        title = "Multivariate Gaussian Mixture Model",
        description = "This example implements variational Bayesian inference in a multivariate Gaussian mixture model with mean-field assumption.",
        hidden = false,
    ),
    (
        path  = "Smoothing Benchmark.ipynb", 
        title = "Smoothing Benchmark", 
        description = "Smoothing Benchmark for Internal testing.",
        hidden = true
    )
]