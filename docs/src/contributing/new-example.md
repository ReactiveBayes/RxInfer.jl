# [Contributing: new example](@id contributing-new-example)

We welcome all possible contributors. This page details the some of the guidelines that should be followed when adding a new example (in the `examples/` folder) to this package.

In order to add a new example simply create a new Jupyter notebook with your experiments in the `examples/` folder. When creating a new example add a descriptive explanation of your experiments, model specification, inference constraints decisions and add appropriate results analysis. For other people it also would be useful if you write descriptive comments along your code.

After that it is necessary to modify the `examples/.meta.jl` file. See the comments in this file for more information.

1. Make sure that the very first cell of the notebook contains ONLY `# <title>` in it and has markdown type. This is important for link generation in the documentation
2. Paths must be local and cannot be located in subfolders
3. Description is used to pre-generate an Examples page overview in the documentation
4. Use hidden option to not include a certain example in the documentation (build will still run to ensure the example runs)
5. Name `Overview` is reserved, please do not use it
6. Use \$\$\begin{aligned} (note the same line, otherwise formulas will not render correctly in the documentation)
                   <latex formulas here>
                   \end{aligned}\$\$ (on the same line (check other examples if you are not sure)
7. Notebooks and plain Julia have different scoping rules for global variables, if it happens so that examples generation fails due to `UndefVarError` and scoping issues use `let ... end` blocks to enforce local scoping (see `Gaussian Mixtures Multivariate.ipynb` as an example)

8. All examples must use and activate local `Project.toml` in the second cell (see `1.`), if you need some package add it to the `(examples)` project

!!! note
    Use `make examples` to run all examples or `make examples specific=MyNewCoolNotebook` to run any notebook that includes `MyNewCoolNotebook` in its file name.