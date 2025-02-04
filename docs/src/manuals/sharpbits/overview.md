# Sharp bits of RxInfer

This page serves as a collection of sharp bits - potential pitfalls and common issues you might encounter while using RxInfer. While RxInfer is designed to be user-friendly, there are certain scenarios where you might encounter unexpected behavior or errors. Understanding these can help you avoid common problems and debug your code more effectively.

- [Rule Not Found Error](@ref rule-not-found)
    - What causes it
    - How to diagnose and fix it
    - Common scenarios

- [Stack Overflow during inference](@ref stack-overflow-inference)
    - Understanding the potential cause
    - Prevention strategies

- [Using `=` instead of `:=` for deterministic nodes](@ref usage-colon-equality)
    - Why not `=`?

- [Getting Help with Issues](@ref getting-help)
    - Using session IDs for better support
    - Understanding telemetry benefits
    - Sharing sessions for debugging

!!! note
    This is a community document that will be updated as we identify more common issues and their solutions. If you encounter a problem that isn't covered here, please consider opening an [issue/discussion](https://github.com/rxinfer/rxinfer/discussions) or contributing to this guide.

## [Getting Help with Issues](@id getting-help)

When you encounter issues with RxInfer, we want to help you as effectively as possible. Here's how you can help us help you:

### Session Data and Telemetry

RxInfer includes optional telemetry and session sharing features to help us provide better support and improve the package. When enabled, these features help us:

- Understand how the package is used in practice
- Identify and debug issues more effectively 
- Make informed decisions about improvements
- Share aggregate usage patterns in community meetings

For details on enabling telemetry and sharing session data, see:
- [Usage Telemetry](@ref manual-usage-telemetry) - Learn about anonymous usage statistics
- [Session Sharing](@ref manual-session-sharing) - Share session data for debugging

All data sharing is optional and privacy-focused. When opening issues, including your session ID helps us provide more targeted support by understanding your usage context.

## How to contribute

If you have a sharp bit to share, please consider opening an [issue/discussion](https://github.com/rxinfer/rxinfer/discussions) or contributing to this guide.
To write a new section, create a new file in the `docs/src/manuals/sharpbits` directory. Use `@id` to specify the ID of the section and `@ref` to reference it later.

```md
# [New section](@id new-section)

This is a new section.
```

Then add a new entry to the `pages` array in the `docs/make.jl` file.

```julia
"Sharp bits of RxInfer" => [
    "Overview" => "manuals/sharpbits/overview.md",
    "Rule Not Found Error" => "manuals/sharpbits/rule-not-found.md",
    "Stack Overflow in Message Computations" => "manuals/sharpbits/stack-overflow-inference.md",
    "Using `=` instead of `:=` for deterministic nodes" => "manuals/sharpbits/usage-colon-equality.md",
    # ...
    "New section" => "manuals/sharpbits/new-section.md",
]
```

In the `overview.md` file, add a new section with the title and the ID of the section. Use the `@ref` macro to reference the ID.

```md
- [New section](@ref new-section)
    - What this section is about
    - ...
```
