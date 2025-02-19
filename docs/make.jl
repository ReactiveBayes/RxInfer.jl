using RxInfer
using Documenter
using DocumenterMermaid
using Dates

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
            "assets/chat.js",
            "assets/favicon.ico"
        ],
        analytics = "G-X4PH160GMF",
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
            "Session summary"           => "manuals/session_summary.md",
            "Sharing sessions & telemetry" => "manuals/telemetry.md",
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

# Function to inject keywords meta tag into HTML files
function inject_keywords_meta()
    # Define keywords for the documentation
    keywords = "Julia, Bayesian inference, factor graph, message passing, probabilistic programming, reactive programming, RxInfer"
    base_url = "https://reactivebayes.github.io/RxInfer.jl/stable"
    
    # Meta tags to inject
    meta_tags = """
    <meta name="keywords" content="$(keywords)">
    <link rel="sitemap" type="application/xml" title="Sitemap" href="$(base_url)/sitemap.xml">"""
    
    # List of files to exclude
    exclude_files = ["googlef2b9004e34bc8cf4.html"]
    
    # Process all HTML files in the build directory
    for (root, _, files) in walkdir("docs/build")
        for file in files
            if endswith(file, ".html") && !(file in exclude_files)
                filepath = joinpath(root, file)
                content = read(filepath, String)
                
                # Insert meta tags before </head>
                new_content = replace(content, r"</head>" => meta_tags * "</head>")
                
                # Write the modified content back
                write(filepath, new_content)
            end
        end
    end
end

# Function to generate sitemap.xml
function generate_sitemap()
    base_url = "https://reactivebayes.github.io/RxInfer.jl"
    sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">"""
    
    # List of files to exclude
    exclude_files = ["googlef2b9004e34bc8cf4.html", "404.html", "search_index.js"]
    
    # Current date in W3C format
    current_date = Dates.format(Dates.now(), "yyyy-mm-dd")
    
    # Process all HTML files in the build directory
    for (root, _, files) in walkdir("docs/build")
        for file in files
            if endswith(file, ".html") && !(file in exclude_files)
                # Get relative path from build directory
                rel_path = relpath(joinpath(root, file), "docs/build")
                # Convert Windows path separators if present
                url_path = replace(rel_path, '\\' => '/')
                # Remove index.html from the end if present
                url_path = replace(url_path, r"/index.html$" => "/")
                # Remove .html from the end of all other files
                url_path = replace(url_path, r"\.html$" => "")
                
                # Construct full URL
                full_url = string(base_url, "/", url_path)
                
                # Add URL entry to sitemap
                sitemap_content *= """
    <url>
        <loc>$(full_url)</loc>
        <lastmod>$(current_date)</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>"""
            end
        end
    end
    
    sitemap_content *= "\n</urlset>"
    
    # Write sitemap to the build directory
    write("docs/build/sitemap.xml", sitemap_content)
end

# Call the functions after makedocs
inject_keywords_meta()
generate_sitemap()

deploydocs(; repo = "github.com/ReactiveBayes/RxInfer.jl", devbranch = "main", forcepush = true)
