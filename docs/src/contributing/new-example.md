# [Contributing: new example](@id contributing-new-example)

We welcome all possible contributors. This page details the some of the guidelines that should be followed when adding a new example (in the `examples/` folder) to this package.

In order to add a new example simply create a new Jupyter notebook with your experiments in the `examples/` folder. When creating a new example add a descriptive explanation of your experiments, model specification, inference constraints decisions and add approriate results analysis. For other people it also would be useful if you write descriptive comments along your code.

After that it is necessary to modify the `examples/.meta.jl` file. See the comments in this file for more information.

!!! note
    Use `make examples` to run all examples or `make examples specific=MyNewCoolNotebook` to run any notebook that includes `MyNewCoolNotebook` in its file name.