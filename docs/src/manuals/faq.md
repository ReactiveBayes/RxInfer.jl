# [Frequently Asked Questions (FAQ)](@id user-guide-faq)

This section addresses common questions and issues that users encounter when working with RxInfer. The FAQ is a living document that grows based on community feedback and common usage patterns.

## General Questions

### What is RxInfer?

RxInfer is a Julia package for automated Bayesian inference on factor graphs using reactive message passing. It provides an efficient, scalable framework for probabilistic programming with a focus on streaming data.

### How does RxInfer compare to other probabilistic programming packages?

See our detailed [comparison guide](@ref comparison) for a comprehensive analysis of RxInfer vs. other tools like Turing.jl, Stan, and PyMC.

### Is RxInfer suitable for beginners?

Yes! RxInfer provides a user-friendly syntax through GraphPPL and comprehensive documentation. Start with the [Getting Started](@ref user-guide-getting-started) guide and work through examples.

## Installation and Setup

See [Installation](@ref user-guide-getting-started-installation) for details.

### I'm getting dependency conflicts. What should I do?

Try `Pkg.resolve()` to resolve conflicts. See [Pkg.jl](https://pkgdocs.julialang.org/v1/getting-started/) for more details.

### Can I use RxInfer from Python?

Yes! See our guide on [Using RxInfer from Python](@ref python-usage).

## Model Specification

### What's the difference between `=` and `:=` in model specification?

- `=` is a regular Julia assignment operator, use it only for regular Julia variables
- `:=` creates a random variable node, use it to create latent variables in your model

See [Sharp Bits: Using `=` instead of `:=`](@ref usage-colon-equality) for details.

### How do I handle missing/incomplete data?

RxInfer supports missing data through the `missing` value in Julia. The inference engine will automatically handle missing observations.
See [Missing Data](@ref manual-static-inference-missing-data) for details.

### How do I create custom nodes and message update rules?

See [Custom Node and Rules](@ref create-node) for detailed guidance on extending RxInfer.

## Inference Issues

### I'm getting "Rule not found" errors. What does this mean?

This error occurs when RxInfer can't find appropriate message update rules for your model. See [Rule Not Found Error](@ref rule-not-found) for solutions.

### "Stack overflow in inference"

See [Stack Overflow during inference](@ref stack-overflow-inference) for more details.

### My inference is running very slowly. How can I improve performance?

Check our [Performance Tips](@ref user-guide-performance-tips) section for optimization strategies.

### How do I debug inference problems?

Check out the [Debugging](@ref user-guide-debugging) guide.

## Performance and Scaling

### How large can my models be?

RxInfer can handle models with millions of latent variables. Performance depends on:
- Model complexity (simple models with conjugate pairs of distributions are the fastest)
- Available memory (large models require more memory)
- Computational resources (more cores, more memory, faster CPU, etc.)
- Optimization techniques used (see [Performance Tips](@ref user-guide-performance-tips))

### Can I use RxInfer for real-time applications?

Yes! RxInfer is designed for real-time inference with reactive message passing. See our [streaming inference](@ref manual-online-inference) documentation.

## Community and Support

### Where can I get help?

1. **Documentation**: Start with the relevant sections
2. **GitHub Discussions**: Ask questions and share experiences
3. **Issues**: Report bugs and request features
4. **Community Meetings**: Join regular public discussions

### How can I contribute?

See our [Contributing Guide](@ref contributing-guidelines) for ways to help. Any help is welcome!

## Contributing to the FAQ

This FAQ grows through community contributions! If you have questions that aren't covered here:

1. **Check existing discussions** on GitHub
2. **Ask your question** in GitHub Discussions
3. **Consider contributing** the answer back to this FAQ
4. **Open an issue** if you find a documentation gap

### How to add questions to the FAQ

1. Open a discussion or issue with your question
2. If it's a common question, consider adding it here
3. Follow the [Contributing to Documentation](@ref guide-docs-contributing) guide
4. Use clear, concise language and include code examples when helpful

---

**Note**: This FAQ is maintained by the community. For the most up-to-date information, check GitHub discussions and issues. If you find outdated information, please help us keep it current!
