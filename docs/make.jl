using RxInfer
using Documenter

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

makedocs(;
    modules = [RxInfer],
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
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/biaslab/RxInfer.jl",
    devbranch = "main"
)
