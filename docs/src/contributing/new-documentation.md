# [Contributing to the documentation](@id guide-docs-contributing)

Contributing to our documentation is a valuable way to enhance the RxInfer ecosystem. To get started, you can follow these steps:

1. **Familiarize Yourself**: First, take some time to explore our existing documentation. Understand the structure, style, and content to align your contributions with our standards.

2. **Identify Needs**: Identify areas that require improvement, clarification, or expansion. These could be missing explanations, code examples, or outdated information.

3. **Fork the Repository**: Fork our documentation repository on GitHub to create your own copy. This allows you to work on your changes independently.

4. **Make Your Edits**: Create or modify content in your forked repository. Ensure your contributions are clear, concise, and well-structured.

5. **Submit a Pull Request**: When you're satisfied with your changes, submit a pull request (PR) to our main repository. Describe your changes in detail in the PR description.

6. **Review and Feedback**: Our documentation maintainers will review your PR. They may provide feedback or request adjustments. Be responsive to this feedback to facilitate the merging process.

7. **Merging**: Once your changes align with our documentation standards, they will be merged into the main documentation. Congratulations, you've successfully contributed to the RxInfer ecosystem!

By following these steps, you can play an essential role in improving and expanding our documentation, making it more accessible and valuable to the RxInfer community.

## Use [`LiveServer.jl`](https://github.com/tlienart/LiveServer.jl)

`LiveServer.jl` is a simple and lightweight web server developed in Julia. It features live-reload capabilities, making it a valuable tool for automatically refreshing the documentation of a package while you work on its content.

To use LiveServer.jl, simply follow these steps[^1]

[^1]: Make sure to install the `LiveServer` and `Documenter` in your current working environment.

- Make sure to import the required packages 
```julia
julia> using LiveServer, Documenter
```

- After importing the required packages, you can start the live server with the following command:
```julia
julia> servedocs()
```

