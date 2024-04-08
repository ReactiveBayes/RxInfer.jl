using RxInfer
using Documenter

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(RxInfer, :DocTestSetup, :(using RxInfer); recursive = true)

# This must be auto-generated with `make examples`
ExamplesPath = joinpath(@__DIR__, "src", "examples")
ExamplesOverviewPath = joinpath(ExamplesPath, "overview.md")
ExamplesMeta = include(joinpath(@__DIR__, "..", "examples", ".meta.jl"))

ExamplesCategories = ExamplesMeta[:categories]
ExamplesCategories = ExamplesCategories[setdiff(keys(ExamplesCategories), [:hidden_examples])]

ExamplesCategoriesOverviewPaths = map(collect(pairs(ExamplesCategories))) do (label, _)
    return joinpath(ExamplesPath, string(label), "overview.md")
end

Examples = map(filter(example -> !isequal(example[:category], :hidden_examples), ExamplesMeta[:examples])) do examplemeta
    filename = examplemeta.filename
    category = examplemeta.category
    mdpath = replace(filename, ".ipynb" => ".md")
    title = examplemeta.title

    fullpath = joinpath(ExamplesPath, string(category), mdpath)
    shortpath = joinpath("examples", string(category), mdpath)

    # We use `fullpath` to check if the file exists
    # We use `shortpath` to make a page reference in the left panel in the documentation
    return title => (fullpath, shortpath, category)
end

if !isdir(ExamplesPath)
    mkpath(ExamplesPath)
end

# Check if some examples are missing from the build
# The `isfile` check needs only the full path, so we ignore the short path
ExistingExamples = filter(Examples) do (title, info)
    fullpath, _, _ = info
    exists = isfile(fullpath)
    if !exists
        @warn "Example at path $(fullpath) does not exist. Skipping."
    end
    return exists
end

# Create an array of pages for each category
ExamplesCategoriesPages = map(collect(pairs(ExamplesCategories))) do (label, category)
    return label => (title = category.title, pages = ["Overview" => joinpath("examples", string(label), "overview.md")])
end |> NamedTuple

# The `pages` argument in the `makedocs` needs only a short path, so we ignore the full path
foreach(ExistingExamples) do (title, info)
    _, shortpath, category = info
    push!(ExamplesCategoriesPages[category].pages, title => shortpath)
end

if length(Examples) !== length(ExistingExamples)
    @warn "Some examples were not found. Use the `make examples` command to generate all examples."
end

# Check if the main `overview.md` file exists + sub categories `overview.md` files
foreach(vcat(ExamplesOverviewPath, ExamplesCategoriesOverviewPaths)) do path
    if !isfile(path)
        @warn "`$(path)` does not exist. Generating an empty overview. Use the `make examples` command to generate the overview and all examples."
        mkpath(dirname(path))
        open(path, "w") do f
            write(f, "The overview is missing. Use the `make examples` command to generate the overview and all examples.")
        end
    end
end

# Generate the final list of examples for each sub-category
ExamplesPages = map(collect(pairs(ExamplesCategoriesPages))) do (label, info)
    return info.title => info.pages
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
    warnonly = Documenter.except(:doctest, :eval_block, :example_block, :meta_block, :parse_error, :setup_block),
    modules = [RxInfer],
    authors = "Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    sitename = "RxInfer.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://reactivebayes.github.io/RxInfer.jl",
        edit_link = "main",
        assets = String["assets/theme.css", "assets/header.css", "assets/header.js"]
    ),
    pages = [
        "Home" => "index.md",
        "User guide" => [
            # "Background: variational inference" => "manuals/background.md",
            "Getting started"           => "manuals/getting-started.md",
            "RxInfer.jl vs. Others"     => "manuals/comparison.md",
            "Model specification"       => "manuals/model-specification.md",
            "Constraints specification" => "manuals/constraints-specification.md",
            "Meta specification"        => "manuals/meta-specification.md",
            "Inference specification"   => ["Overview" => "manuals/inference/overview.md", "Static vs Streamline inference" => "manuals/inference/infer.md", "Inference results postprocessing" => "manuals/inference/postprocess.md", "Manual inference specification" => "manuals/inference/manual.md"],
            "Inference customization"   => ["Defining a custom node and rules" => "manuals/custom-node.md"],
            "Messages initialization"   => "manuals/understanding-why-to-initialize-messages.md",
            "Debugging"                 => "manuals/debugging.md",
            "Delta node"                => "manuals/delta-node.md"
        ],
        "Library" => [
            "Built-in functional form constraints" => "library/functional-forms.md",
            "Model specification" => "library/model-specification.md",
            "Bethe Free Energy" => "library/bethe-free-energy.md",
            "Exported methods" => "library/exported-methods.md"
        ],
        "Examples" => [
            "Overview" => "examples/overview.md", # This must be auto-generated with `make examples`
            ExamplesPages...,
            "External examples" => "contributing/external-examples.md"
        ],
        "Contributing" => [
            "Contribution guide" => "contributing/guide.md",
            "Contribution guidelines"=> "contributing/guidelines.md",
            "Contributing to the documentation" => "contributing/new-documentation.md",
            "Contributing to the examples" => "contributing/new-example.md",
            "Publishing a new release" => "contributing/new-release.md"
        ]
    ]
)

deploydocs(; repo = "github.com/ReactiveBayes/RxInfer.jl", devbranch = "main", forcepush = true)
