using RxInfer
using Documenter

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive=true)

makedocs(;
    modules=[RxInfer],
    authors="Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    repo="https://github.com/bvdmitri/RxInfer.jl/blob/{commit}{path}#{line}",
    sitename="RxInfer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bvdmitri.github.io/RxInfer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bvdmitri/RxInfer.jl",
    devbranch="main",
)
