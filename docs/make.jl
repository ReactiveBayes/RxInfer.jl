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
        "Introduction to RxInfer" => [
            "Background: variational inference" => "intro/inference/background.md",
            "Inference for static datasets"     => "intro/inference/inference.md",
            "Inference for real-time datasets"  => "intro/inference/rxinference.md"
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
