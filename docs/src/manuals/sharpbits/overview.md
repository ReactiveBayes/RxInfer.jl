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

!!! note
    This is a community document that will be updated as we identify more common issues and their solutions. If you encounter a problem that isn't covered here, please consider opening an [issue/discussion](https://github.com/rxinfer/rxinfer/discussions) or contributing to this guide.

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
