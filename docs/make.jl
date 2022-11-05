using RxInfer
using Documenter

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

# This must be auto-generated with `make examples`
ExamplesPath = joinpath(@__DIR__, "src", "examples")
Examples = map(filter((f) -> f != "Overview.md", readdir(ExamplesPath))) do example 
    return replace(example, ".md" => "") => joinpath("examples", example)
end

# WIP: Keep it as a nice starting approach for adding a header, currently we are using `assets/header.js`
# struct DocumentationWriter <: Documenter.Writer
#     base :: Documenter.HTML
# end

# abstract type ExtendedHTMLFormat <: Documenter.Writers.FormatSelector end

# Documenter.Selectors.order(::Type{ExtendedHTMLFormat})            = 4.0
# Documenter.Selectors.matcher(::Type{ExtendedHTMLFormat}, fmt, _)  = isa(fmt, DocumentationWriter)

# function Documenter.Selectors.runner(::Type{ExtendedHTMLFormat}, fmt, doc) 
#     return Documenter.Writers.HTMLWriter.render(doc, fmt.base)
# end

makedocs(;
    draft = false,
    strict = [ :doctest, :eval_block, :example_block, :meta_block, :parse_error, :setup_block ],
    modules = [ RxInfer ],
    authors = "Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    repo = "https://github.com/biaslab/RxInfer.jl/blob/{commit}{path}#{line}",
    sitename = "RxInfer.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://biaslab.github.io/RxInfer.jl",
        edit_link = "main",
        assets = String[ "assets/theme.css", "assets/header.css", "assets/header.js" ],
    ),
    pages = [
        "Home"     => "index.md",
        "User guide"  => [
            # "Background: variational inference" => "manuals/background.md",
            "Getting started"           => "manuals/getting-started.md",
            "Model specification"       => "manuals/model-specification.md",
            "Constraints specification" => "manuals/constraints-specification.md",
            "Meta specification"        => "manuals/meta-specification.md",
            "Inference specification"   => [ 
                "Overview" => "manuals/inference/overview.md",
                "Static dataset" => "manuals/inference/inference.md",
                "Real-time dataset / reactive inference" => "manuals/inference/rxinference.md",
                "Manual inference specification" => "manuals/inference/manual.md"
            ]
        ],
        "Library" => [
            "Built-in functional form constraints" => "library/functional-forms.md",
            "Model specification" => "library/model-specification.md",
            "Bethe Free Energy" => "library/bethe-free-energy.md",
            "Exported methods" => "library/exported-methods.md"
        ],
        "Examples" => [
            "Overview" => "examples/overview.md", # This must be auto-generated with `make examples`
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
