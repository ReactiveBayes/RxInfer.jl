using RxInfer
using Documenter

## Custom theme building

using Sass

assets = joinpath(@__DIR__, "src", "assets")
ispath(assets) || mkpath(assets)

themesrc = joinpath(@__DIR__, "theme", "theme.scss")
themedst = joinpath(assets, "theme.css")
    
Sass.compile_file(themesrc, themedst)

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

# This must be auto-generated with `make examples`
ExamplesPath = joinpath(@__DIR__, "src", "examples")
Examples = map(filter((f) -> f != "Overview.md", readdir(ExamplesPath))) do example 
    return replace(example, ".md" => "") => joinpath("examples", example)
end

# struct DocumentationWriter <: Documenter.Writer
#     base :: Documenter.HTML
# end

# function Documenter.Selectors.runner(::Type{T}, writer::DocumentationWriter, document) where {T}
#     Documenter.Selectors.runner(T, writer.base, document)
# end

makedocs(;
    draft = true,
    modules = [ RxInfer ],
    authors = "Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    repo = "https://github.com/biaslab/RxInfer.jl/blob/{commit}{path}#{line}",
    sitename = "RxInfer.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://biaslab.github.io/RxInfer.jl",
        edit_link = "main",
        assets = String[ "assets/theme.css" ],
        footer = """
        <b>Privet</b>
        """
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
            "Model specification" => "library/model-specification.md",
            "Bethe Free Energy" => "library/bethe-free-energy.md",
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
