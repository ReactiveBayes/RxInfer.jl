SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: lint format

scripts_init:
	julia --startup-file=no --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.precompile();'

lint: scripts_init ## Code formating check
	julia --startup-file=no --project=scripts/ scripts/format.jl

format: scripts_init ## Code formating run
	julia --startup-file=no --project=scripts/ scripts/format.jl --overwrite

.PHONY: examples

examples_init:
	rm -f examples/Manifest.toml
	julia --startup-file=no --project=examples/ -e 'ENV["PYTHON"]=""; using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.precompile(); Pkg.build("PyPlot"); using Plots; using PyPlot'

dev_examples_init:
	rm -f examples/Manifest.toml
	julia --startup-file=no --project=examples/ -e 'ENV["PYTHON"]=""; using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "ReactiveMP.jl"))); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "GraphPPL.jl"))); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "Rocket.jl"))); Pkg.update(); Pkg.precompile(); Pkg.build("PyPlot"); using Plots; using PyPlot'

examples: scripts_init examples_init ## Precompile examples and put them in the `docs/src/examples` folder (use specific="<pattern>" to compile a specific example)
	julia --startup-file=no --project=scripts/ scripts/examples.jl $(specific)

devexamples: scripts_init dev_examples_init ## Same as `make examples` but uses `dev-ed` versions of core packages
	julia --startup-file=no --project=scripts/ scripts/examples.jl $(specific)

.PHONY: docs

doc_init:
	rm -f docs/Manifest.toml
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.precompile();'

dev_doc_init:
	rm -f docs/Manifest.toml
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "ReactiveMP.jl"))); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "GraphPPL.jl"))); Pkg.develop(PackageSpec(path=joinpath(Pkg.devdir(), "Rocket.jl"))); Pkg.update(); Pkg.precompile();'

docs: doc_init ## Generate documentation
	julia --startup-file=no --project=docs/ docs/make.jl

devdocs: dev_doc_init ## Same as `make docs` but uses `dev-ed` versions of core packages
	julia --startup-file=no --project=docs/ docs/make.jl

.PHONY: test

test: ## Run tests, use test_args="folder1:test1 folder2:test2" argument to run reduced testset, use dev=true to use `dev-ed` version of core packages
	julia -e 'ENV["USE_DEV"]="$(dev)"; import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'	

devtest: ## Alias for the `make test dev=true ...`
	julia -e 'ENV["USE_DEV"]="true"; import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'	

clean: ## Clean documentation build, precompiled examples, benchmark output from tests
	rm -rf docs/src/examples
	rm -rf docs/src/assets/examples
	rm -rf docs/build
	rm -rf _output
	rm -rf test/_output
	
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
