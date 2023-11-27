# [Contributing to the examples](@id contributing-new-example)

We welcome all possible contributors. This page details some of the guidelines that should be followed when adding a new example (in the `examples/` folder) to this package.

In order to add a new example simply create a new Jupyter notebook with your experiments in the `examples/"subcategory"` folder. When creating a new example add a descriptive explanation of your experiments, model specification, inference constraints decisions and add appropriate results analysis. We expect examples to be readable for the general public and therefore highly value descriptive comments. If a submitted example only contains code, we will kindly request some changes to improve the readability.

After preparing the new example it is necessary to modify the `examples/.meta.jl` file.

1. Make sure that the very first cell of the notebook contains ONLY `# <title>` in it and has the markdown cell type. This is important for generating links in our documentation.
2. The `path` option must be set to a local path in a category sub-folder.
3. The text in the `description` option will be used on the `Examples` page in the documentation.
4. Set `category = :hidden_examples` to hide a certain example in the documentation (the example will be executed to ensure it runs without errors).
5. Please do no use `Overview` as a name for the new example, the title `Overview` is reserved.
6. Use the following template for equations, note that `$$` and both `\begin` and `\end` commands are on the same line (check other examples if you are not sure). This is important, because otherwise formulas may not render correctly. Inline equations may use `$...$` template.
```
$$\begin{aligned}
      <latex equations here>
\end{aligned}$$
``` 
7. When using equations, make sure not to follow the left-hand `$$` or `$` with a space, but instead directly start the equation, e.g. not `$$ a + b $$`, but `$$a + b$$`. For equations that are supposed to be on a separate line, make sure `$$...$$` is preceded and followed by an empty line.
8. Notebooks and plain Julia have different scoping rules for global variables. It may happen that the generation of your example fails due to an `UndefVarError` or other scoping issues. In these cases we recommend using `let ... end` blocks to enforce local scoping or use the `global` keyword to disambiguate the scoping rules, e.g.
```julia
variable = 0
for i in 1:10
    global variable = variable + i
end
```
9. All examples must use and activate the local environment specified by `Project.toml` in the second cell (see `1.`). Please have a look at the existing notebooks for an example on how to activate this local environment. If you need additional packages, you can add then to the `(examples)` project.
10. All plots should be displayed automatically. In special cases, if needed, save figures in the `../pics/figure-name.ext` format. Might be useful for saving gifs. Use `![](../pics/figure-name.ext)` to display a static image.

!!! note
    Please avoid adding `PyPlot` in the `(examples)` project. Installing and building `PyPlot` dependencies takes several minutes on every CI run. Use `Plots` instead.

!!! note
    Use `make examples` to run all examples or `make examples specific=MyNewCoolNotebook` to run any notebook that includes `MyNewCoolNotebook` in its file name.
