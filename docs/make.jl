using RxInfer
using Documenter
using DocumenterMermaid

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

draft = get(ENV, "DOCS_DRAFT", "false") == "true"

makedocs(;
    draft = draft,
    warnonly = false,
    modules = [RxInfer],
    authors = "Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    sitename = "RxInfer.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://reactivebayes.github.io/RxInfer.jl",
        edit_link = "main",
        warn_outdated = true,
        assets = [
            "assets/theme.css",
            "assets/header.css",
            "assets/header.js",
            asset("https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/themes/df-messenger-default.css"),
            asset("https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/df-messenger.js"),
            "assets/chat.js"
        ],
        description = "Julia package for automated Bayesian inference on a factor graph with reactive message passing",
        footer = "Created in [BIASlab](https://biaslab.github.io/), maintained by [ReactiveBayes](https://github.com/ReactiveBayes), powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and the [Julia Programming Language](https://julialang.org/)."
    ),
    pages = [
        "Home" => "index.md",
        "User guide" => [
            "Getting started"           => "manuals/getting-started.md",
            "RxInfer.jl vs. Others"     => "manuals/comparison.md",
            "Model specification"       => "manuals/model-specification.md",
            "Constraints specification" => "manuals/constraints-specification.md",
            "Meta specification"        => "manuals/meta-specification.md",
            "Inference specification"   => ["Overview" => "manuals/inference/overview.md", "Static inference" => "manuals/inference/static.md", "Streamline inference" => "manuals/inference/streamlined.md", "Initialization" => "manuals/inference/initialization.md", "Auto-updates" => "manuals/inference/autoupdates.md", "Deterministic nodes" => "manuals/inference/delta-node.md", "Non-conjugate inference" => "manuals/inference/nonconjugate.md", "Undefined message update rules" => "manuals/inference/undefinedrules.md"],
            "Inference customization"   => ["Defining a custom node and rules" => "manuals/customization/custom-node.md", "Inference results postprocessing" => "manuals/customization/postprocess.md"],
            "Debugging"                 => "manuals/debugging.md",
            "Migration from v2 to v3"   => "manuals/migration-guide-v2-v3.md",
            "Sharp bits of RxInfer"     => ["Overview" => "manuals/sharpbits/overview.md", "Rule Not Found Error" => "manuals/sharpbits/rule-not-found.md", "Stack Overflow in Message Computations" => "manuals/sharpbits/stack-overflow-inference.md", "Using `=` instead of `:=` for deterministic nodes" => "manuals/sharpbits/usage-colon-equality.md"]
        ],
        "Library" => [
            "Model construction" => "library/model-construction.md",
            "Bethe Free Energy" => "library/bethe-free-energy.md",
            "Functional form constraints" => "library/functional-forms.md",
            "Exported methods" => "library/exported-methods.md"
        ],
        "Examples" => "examples/overview.md",
        "Contributing" => [
            "Contribution guide" => "contributing/guide.md",
            "Contribution guidelines" => "contributing/guidelines.md",
            "Contributing to the documentation" => "contributing/new-documentation.md",
            "Contributing to the examples" => "contributing/new-example.md",
            "Publishing a new release" => "contributing/new-release.md"
        ]
    ]
)

deploydocs(; repo = "github.com/ReactiveBayes/RxInfer.jl", devbranch = "main", forcepush = true)
