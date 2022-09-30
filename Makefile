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
	julia --startup-file=no --project=examples/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.precompile(); using Plots; using PyPlot'

examples: scripts_init examples_init ## Precompile examples and put them in the `docs/src/examples` folder (use specific="<pattern>" to compile a specific example)
	julia --startup-file=no --project=scripts/ scripts/examples.jl $(specific)

.PHONY: docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.precompile();'

docs: doc_init ## Generate documentation
	julia --startup-file=no --project=docs/ docs/make.jl

.PHONY: test

test: ## Run tests (use testset="folder1:test1 folder2:test2" argument to run reduced testset)
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(testset)") .|> string)'	

clean: ## Clean documentation build, precompiled examples, benchmark output from tests
	rm -rf docs/src/examples
	rm -rf docs/src/assets/examples
	rm -rf docs/build
	rm -rf _output
	rm -rf test/_output
	
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)