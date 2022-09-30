using RxInfer
using Documenter

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

# This must be auto-generated with `make examples`
ExamplesPath = joinpath(@__DIR__, "src", "examples")
Examples = map(filter((f) -> f != "Overview.md", readdir(ExamplesPath))) do example 
    return replace(example, ".md" => "") => joinpath("examples", example)
end

makedocs(;
    modules = [ RxInfer ],
    authors = "Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    repo = "https://github.com/biaslab/RxInfer.jl/blob/{commit}{path}#{line}",
    sitename = "RxInfer.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://biaslab.github.io/RxInfer.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home"     => "index.md",
        "User guide"  => [
            "Background: variational inference" => "manuals/background.md",
            "Getting started"           => "manuals/getting-started.md",
            "Model specification"       => "manuals/model-specification.md",
            "Constraints specification" => "manuals/constraints-specification.md",
            "Meta specification"        => "manuals/meta-specification.md",
            "Inference execution"       => "manuals/inference-execution.md"
        ],
        "Library" => [
            "Built-in functional form constraints" => "library/functional-forms.md",
            "Exported methods" => "library/exported-methods.md"
        ],
        "Examples" => [
            "Overview" => "examples/Overview.md", # This must be auto-generated with `make examples`
            Examples...
        ],
        "Contributing" => [
            "Overview" => "contributing/overview.md",
            "Adding a new example" => "contributing/new-example.md",
        ]
    ]
)

deploydocs(;
    repo = "github.com/biaslab/RxInfer.jl",
    devbranch = "main"
)
