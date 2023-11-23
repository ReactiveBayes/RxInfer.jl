# Contributing to the dependencies

Julia programming language makes it extremely easy to create, develop and register new packages in the ecosystem.

## The benefits of small packages for the ecosystem

In the world of software development, there's often a choice to be made between creating a single monolithic package or breaking your codebase into smaller, more focused packages. While both approaches have their merits, opting for smaller packages can offer several significant benefits:

1. **Modularity**: Smaller packages focus on specific tasks, making them easier to maintain and debug.

2. **Collaboration**: Teams can work on different packages concurrently, speeding up development.

3. **Version Control**: Precise versioning and fewer dependencies lead to leaner projects.

4. **Performance**: Smaller packages can result in faster precompilation (in Julia) and more efficient testing.

5. **Flexibility**: Developers can select and customize packages for their needs.

6. **Community**: Smaller packages attract contributors, fostering collaboration and faster feedback.


In summary, while monolithic packages have their place, opting for smaller, focused packages can bring numerous advantages in terms of modularity, collaboration, version control, flexibility, and community engagement.

## Use [`PkgTemplates`](https://github.com/JuliaCI/PkgTemplates.jl)

`PkgTemplates.jl` is a Julia package to create new Julia packages in an easy, repeatable, and customizable way.
You can use the following template to generate a new package:

```julia
julia> using PkgTemplates

julia> USER = "your github user name" # Use `biaslab` if developing within the BIASlab organisation

julia> template = Template(
    user = USER, 
    plugins = [
        CompatHelper(), 
        ProjectFile(), 
        SrcDir(), 
        Git(), 
        License(), 
        Readme(), 
        Tests(), 
        GitHubActions(), 
        Codecov(), 
        Documenter{GitHubActions}(), 
        Formatter(style="blue"), 
        BlueStyleBadge(), 
        PkgEvalBadge()
])

julia> template("MyNewCoolPackage")
```

This template generates a standard Julia package complete with streamlined documentation, tests, code coverage, and [Blue style formatting](https://github.com/invenia/BlueStyle). Refer to the `PkgTemplates` documentation if you wish to customize certain steps in the process.

- #### Adjust the minimum supported version of Julia

After auto-generation, the minimum supported Julia version will be set to `1.0.0`. You can modify this in the `Project.toml` file, for example:

```toml
[compat]
julia = "1.9.2"
```

Try to be conservative and set as low version of Julia as possible.

- #### Adjust the authors of the package

The authors field is present in the `Project.toml`, e.g 

```toml
authors = ["John Wick <john.wick@continental.com>", ...]
```

- #### Add requires dependencies and their `[compat]` bounds

To add new dependencies to your newly created package, start Julia in the package's folder and activate the project using one of the following methods:

```bash
julia --project=.
```

or 

```bash
julia
```

```julia
julia> ] activate .
```

Then, add dependencies like this:

```julia
julia> ] add SomeCoolDependency, SomeOtherCoolDependency
```

For each new dependency, it's essential to specify the minimum compatible version in the `[compat]`section of the `Project.toml` file, otherwise the official Julia registry will not register your new package. Add the `[compat]` entries like this:

```toml
[compat]
julia = "1.9"
SomeCoolDependency = "0.19.2"
SomeOtherCoolDependency = "1.3.12"
```

For more details on `compat` bounds, check the official Julia documentation.

## Adjust `README.md`

The `README.md` file is the front door to your project, offering a concise introduction and guidance for users and contributors. It's a critical piece of documentation that sets the tone for your project's accessibility and success. A well-crafted `README.md` provides essential information, such as installation instructions, usage examples, and project goals, making it easier for others to understand, engage with, and contribute to your work. So, remember, taking the time to write a clear and informative `README.md`.

!!! note
    Some badges in the auto-generated `README.md` will be broken unless you register your package in the official Julia registry.

## Write code and tests

The provided template generates a package with testing and test coverage enabled. Ensure to test all new functionality in the `test/runtests.jl` file.

### Simplify Testing with `ReTestItems`

You can streamline testing by using the `ReTestItems` package, which support VSCode UI for running tests.
Refer to the `ReTestItems` documentation for more information.

## Write code and the documentation

Julia, adding documentation is straightforward with the `Documenter.jl` package. Add docstrings to newly created functions and update the `docs/index.md`  ` file. 
To build the documentation locally, use this command (ensure you initialize and instantiate the docs environment first):

```bash
julia --project=docs docs/make.jl
```

For customization, refer to the `Documenter.jl` documentation. Also, check out the [contributing guide](@ref guide-docs-contributing).

### Hosting Documentation with GitHub Actions

The template provided generates a package with automatic documentation hosting through GitHub Actions. To make this process work, you'll need to generate a `DOCUMENTER_KEY` using `DocumenterTools.jl` and add it to your package's repository settings. You can find detailed instructions on how to do this [here](https://documenter.juliadocs.org/stable/man/hosting/#travis-ssh).

### Enable GitHub Pages in Repository Settings

The final step for setting up documentation hosting is to enable GitHub Pages in your package's repository settings. To do this:

1. Navigate to the GitHub Pages settings of your repository.
2. Choose the **Deploy from a branch** option.
3. Select the `gh-pages` branch.

