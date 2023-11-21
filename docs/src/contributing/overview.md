# [Contributing](@id contributing-overview)

We welcome all possible contributors. This page details some of the guidelines that should be followed when contributing to this package.

## Reporting bugs

We track bugs using [GitHub issues](https://github.com/biaslab/RxInfer.jl/issues). We encourage you to write complete, specific, reproducible bug reports. Mention the versions of Julia and `RxInfer` for which you observe unexpected behavior. Please provide a concise description of the problem and complement it with code snippets, test cases, screenshots, tracebacks or any other information that you consider relevant. This will help us to replicate the problem and narrow the search space for solutions.

### Nightly Julia status

The badge that indicates if `RxInfer` can be installed on a Julia nightly version. The failing badge may indicate either a problem with `RxInfer` itself of with one if the dependencies. 
Click on the badge to get the latest evaluation report.

[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/R/RxInfer.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Suggesting features

We welcome new feature proposals. However, before submitting a feature request, consider a few things:

- Does the feature require changes in the core `RxInfer` code? If it doesn't (for example, you would like to add a factor node for a particular application), you can add local extensions in your script/notebook or consider making a separate repository for your extensions.
- If you would like to add an implementation of a feature that changes a lot in the core `RxInfer` code, please open an issue on GitHub and describe your proposal first. This will allow us to discuss your proposal with you before you invest your time in implementing something that may be difficult to merge later on.

## Contributing code

### Installing RxInfer

We suggest that you use the `dev` command from the new Julia package manager to
install `RxInfer` for development purposes. To work on your fork of `RxInfer`, use your fork's URL address in the `dev` command, for example:

```
] dev git@github.com:your_username/RxInfer.jl.git
```

The `dev` command clones `RxInfer` to `~/.julia/dev/RxInfer`. All local changes to `RxInfer` code will be reflected in imported code.

!!! note
    It is also might be useful to install [Revise.jl](https://github.com/timholy/Revise.jl) package as it allows you to modify code and use the changes without restarting Julia.

### Core dependencies

`RxInfer.jl` heavily depends on the `ReactiveMP.jl`, `GraphPPL.jl` and `Rocket.jl` packages. RxInfer.jl must be updated every time any of these packages has a major update and/or API changes. Developers are adviced to use the `dev` command for all of these packages while making changes to the `RxInfer.jl`. It is worth noting though that standard Julia testing utilities ignore the local development environment and always try to test the package with the latest released versions of the core dependencies. Read the section about the `Makefile` below to see how to test `RxInfer.jl` with the locally installed core dependencies.

### Committing code

We use the standard [GitHub Flow](https://guides.github.com/introduction/flow/) workflow where all contributions are added through pull requests. In order to contribute, first [fork](https://guides.github.com/activities/forking/) the repository, then commit your contributions to your fork, and then create a pull request on the `main` branch of the `RxInfer` repository.

Before opening a pull request, please make sure that all tests pass without failing! All examples (can be found in `/examples/` directory) have to run without errors as well. 

!!! note
    Use `make test`, `make examples` and `make docs` commands to ensure that all tests, examples and the documentation build run without any issues. See below for the `Makefile` commands description in more details.

### Style conventions

We use default [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/index.html). We list here a few important points and our modifications to the Julia style guide:

- Use 4 spaces for indentation
- Type names use `UpperCamelCase`. For example: `AbstractFactorNode`, `RandomVariable`, etc..
- Function names are `lowercase` with underscores, when necessary. For example: `activate!`, `randomvar`, `as_variable`, etc..
- Variable names and function arguments use `snake_case`
- The name of a method that modifies its argument(s) must end in `!`

!!! note
    `RxInfer` repository contains scripts to automatically format code according to our guidelines. Use `make format` command to fix code style. This command overwrites files. Use `make lint` to run a linting procedure without overwriting the actual source files.

### Unit tests

We use the test-driven development (TDD) methodology for `RxInfer` development. The test coverage should be as complete as possible. Please make sure that you write tests for each piece of code that you want to add.

All unit tests are located in the `/test/` directory. The `/test/` directory follows the structure of the `/src/` directory. Each test file should have the following filename format: `test_*.jl`. Some tests are also present in `jldoctest` docs annotations directly in the source code.
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
