# [Contribution Guidelines](@id contributing-overview)

Welcome to the contribution guidelines for `RxInfer.jl`. Here you'll find detailed instructions on how to contribute effectively to the project.

## Reporting bugs

If you encounter any bugs while using the software, please report them via [GitHub issues](https://github.com/reactivebayes/RxInfer.jl/issues). To ensure efficient bug resolution, please provide comprehensive and reproducible bug reports. Include details such as the versions of Julia and `RxInfer` you're using, along with a concise description of the issue. Additionally, attach relevant code snippets, test cases, screenshots, or any other pertinent information that could aid in replicating and addressing the problem.

### Nightly Julia status

Check the badge below to see if `RxInfer` can be installed on a Julia nightly version. A failing badge may indicate issues with `RxInfer` or its dependencies. Click on the badge to access the latest evaluation report.

[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/R/RxInfer.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Suggesting features

We encourage proposals for new features. Before submitting a feature request, consider the following:

- Determine if the feature necessitates changes to the core `RxInfer` code. If not, such as adding a factor node for a specific application, consider local extensions in your script/notebook or creating a separate repository for your extensions.
- For feature implementations requiring significant changes to the core `RxInfer` code, open a [GitHub issue](https://github.com/reactivebayes/RxInfer.jl/issues) to discuss your proposal before proceeding with implementation. This allows for thorough deliberation and avoids investing time in features that may be challenging to integrate later on.

## Contributing code

### Installing RxInfer

For development purposes, it's recommended to use the `dev` command from the Julia package manager to install `RxInfer`. Use your fork's URL in the dev command to work on your forked version. For example:

```
] dev git@github.com:your_username/RxInfer.jl.git
```

The `dev` command clones `RxInfer` to `~/.julia/dev/RxInfer`. All local changes to `RxInfer` code will be reflected in imported code.

!!! note
    It is also might be useful to install [Revise.jl](https://github.com/timholy/Revise.jl) package as it allows you to modify code and use the changes without restarting Julia.

### Core dependencies

`RxInfer.jl` depends heavily on the core packages `ReactiveMP.jl`, `GraphPPL.jl`, and `Rocket.jl`. Ensure `RxInfer.jl` is updated whenever any of these packages undergo major updates or API changes. While making changes to `RxInfer.jl`, developers are advised to use the `dev` command for these packages as well. Note that standard Julia testing utilities ignore the local development environment and test the package with the latest released versions of core dependencies. Refer to the Makefile section below to learn how to test `RxInfer.jl` with locally installed core dependencies.

### Committing code

We use the standard [GitHub Flow](https://guides.github.com/introduction/flow/) workflow where all contributions are added through pull requests. To contribute:
- [Fork](https://guides.github.com/activities/forking/) the repository
- Commit your contributions to your fork
- Create a pull request on the `main` branch of the `RxInfer` repository.

Before opening a pull request, ensure all tests pass without errors. Additionally, ensure all examples (found in the `/examples/` directory) run succesfully. 

!!! note
    Use `make test`, `make examples` and `make docs` commands to verify that all tests, examples, and documentation build correctly. See the `Makefile` section below for detailed command descriptions.

### Style conventions

We use the default [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/index.html). There are a couple of important points modifications to the Julia style guide to take into account:

- Use 4 spaces for indentation
- Type names use `UpperCamelCase`. For example: `AbstractFactorNode`, `RandomVariable`, etc..
- Function names are `lowercase` with underscores, when necessary. For example: `activate!`, `randomvar`, `as_variable`, etc..
- Variable names and function arguments use `snake_case`
- The name of a method that modifies its argument(s) must end in `!`

!!! note
    The `RxInfer` repository contains scripts to automatically format code according to our guidelines. Use `make format` command to fix code style. This command overwrites files. Use `make lint` to run a linting procedure without overwriting the actual source files.

### Unit tests

We use the test-driven development (TDD) methodology for `RxInfer` development. Aim for comprehensive test coverage, ensuring tests cover each piece of added code.

All unit tests are located in the `/test/` directory. The `/test/` directory follows the structure of the `/src/` directory. Each test file should have the following filename format: `*_tests.jl`. Some tests are also present in `jldoctest` docs annotations directly in the source code.
See [Julia's documentation](https://docs.julialang.org/en/v1/manual/documentation/index.html) about doctests.

The tests can be evaluated by running following command in the Julia REPL:

```
] test RxInfer
```

In addition tests can be evaluated by running following command in the `RxInfer` root directory:

```bash
make test
```

!!! note 
    Use `make devtest` to use local `dev`-ed versions of the core packages.

### Makefile

`RxInfer.jl` uses `Makefile` for most common operations:

- `make help`: Shows help snippet
- `make test`: Run tests, supports extra arguments
  - `make test test_args="distributions:normal_mean_variance"` would run tests only from `distributions/test_normal_mean_variance.jl`
  - `make test test_args="distributions:normal_mean_variance models:lgssm"` would run tests both from `distributions/test_normal_mean_variance.jl` and `models/test_lgssm.jl`
  - `make test dev=true` would run tests while using `dev-ed` versions of core packages
- `make devtest`: Alias for the `make test dev=true ...`
- `make docs`: Compile documentation
- `make devdocs`: Same as `make docs`, but uses `dev-ed` versions of core packages
- `make examples`: Run all examples and put them in the `docs/` folder if successfull 
- `make devexamples`: Same as `make examples`, but uses `dev-ed` versions of core packages
- `make lint`: Check codestyle
- `make format`: Check and fix codestyle 

!!! note
    Core packages include `ReactiveMP.jl`, `GraphPPL.jl` and `Rocket.jl`. When using any of the `dev` commands from the `Makefile` those packages must be present in the `Pkg.devdir()` directory.
