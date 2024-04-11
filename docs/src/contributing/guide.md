# [Contributing](@id contributing-overview)
Welcome to the contribution guide for `RxInfer.jl`. Here you'll find information on the `RxInfer` project structure, and how to get started with contributing to the project. For more practical instructions and guidelines, refer to the [contribution guidelines](https://reactivebayes.github.io/RxInfer.jl/stable/contributing/guidelines.html).

## Project structure

`RxInfer.jl` is a Julia package that provides a high-level interface for probabilistic programming. It is composed of three major core dependencies:
- `Rocket.jl`: A package for reactive programming, allowing asynchronous data processing
- `GraphPPL.jl`: A domain-specific language for probabilistic programming, facilitating the `@model` macro and other crucial user-facing features.
- `ReactiveMP.jl`: Reactive message passing engine, using `Rocket.jl` to pass messages between nodes in a probabilistic model defined with `GraphPPL.jl`.

In general, non-inference related functionality is implemented in `Rocket.jl` and `GraphPPL.jl`, while inference-related functionality is implemented in `ReactiveMP.jl`. For example, all factor nodes and inference rules for messages are implemented in `ReactiveMP.jl`.

## Getting started

To familiarize yourself with development in `RxInfer`, we recommend the following steps:
1. Familiarize yourself with the collaborative tools used in the project. `RxInfer` uses GitHub for version control, issue tracking, and pull requests. We aim to maintain the [good first issue](https://github.com/ReactiveBayes/RxInfer.jl/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) label on issues that are suitable for new contributors. Furthermore, the core development team tracks the project's progress and development tasks on the [project board](https://github.com/orgs/ReactiveBayes/projects/2/views/4). Because the project board is overwhelming, we recommend focusing first on issues labeled with the `good first issue` label. 
2. Read the [contribution guidelines](https://reactivebayes.github.io/RxInfer.jl/stable/contributing/guidelines.html) to understand the contribution process and best practices for contributing to `RxInfer`, as well as coding practices and testing procedures.
3. Familiarize yourself with the `RxInfer` codebase and its core dependencies. While most information can be found on the `RxInfer` documentation page, it is also recommended to read the documentation for `Rocket.jl`, `GraphPPL.jl`, and `ReactiveMP.jl` to understand the core functionality and design principles of the project.
4. Pick an issue to work on. We recommend starting with a `good first issue` to familiarize yourself with the contribution process. Once you're comfortable with the process, you can move on to more complex issues.

## Contribution guidelines

The contribution guidelines provide detailed instructions on how to contribute effectively to the project. They cover reporting bugs, suggesting features, and contributing code. For more information, refer to the [contribution guidelines](https://reactivebayes.github.io/RxInfer.jl/stable/contributing/guidelines.html).

