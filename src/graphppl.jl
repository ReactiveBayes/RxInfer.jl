
using GraphPPL, TupleTools, MacroTools

import MacroTools: @capture, postwalk, prewalk, walk

export @model, @constraints, @meta

struct RxInferBackend end

function GraphPPL.write_model_structure(::RxInferBackend, ms_name, ms_model, ms_args_checks, ms_args_const_init_block, ms_args, ms_kwargs, ms_body)
    generator = gensym(ms_name)

    # Extract symbols from the arguments specification
    ms_args_symbols = map(ms_args) do ms_arg
        if @capture(ms_arg, sym_ = var_)
            return sym
        end
        return ms_arg
    end

    # Extract symbols from the keyword arguments specification
    ms_kwargs_symbols = map(ms_kwargs) do ms_kwarg
        if @capture(ms_kwarg, sym_ = var_)
            return sym
        end
        return ms_kwarg
    end

    return quote
        function ($generator)($ms_model::RxInfer.FactorGraphModel, $(ms_args...); $(ms_kwargs...))
            $(ms_args_checks...)
            $(ms_args_const_init_block...)
            $ms_body
        end

        function $ms_name($(ms_args...); $(ms_kwargs...))
            return RxInfer.ModelGenerator($generator, ($(ms_args_symbols...),), (; $(ms_kwargs_symbols...)))
        end

        RxInfer.model_name(::typeof($ms_name)) = $(QuoteNode(ms_name))
    end
end

function GraphPPL.write_argument_guard(::RxInferBackend, argument::Symbol)
    return :(@assert !($argument isa ReactiveMP.AbstractVariable) "It is not allowed to pass AbstractVariable objects to a model definition arguments. ConstVariables should be passed as their raw values.")
end

function GraphPPL.write_randomvar_expression(::RxInferBackend, model, varexp, options, arguments)
    return :($varexp = ReactiveMP.randomvar($model, $options, $(GraphPPL.fquote(varexp)), $(arguments...)); $varexp)
end

function GraphPPL.write_datavar_expression(::RxInferBackend, model, varexpr, options, type, arguments)
    errstr    = "The expression `$varexpr = datavar($(type))` is incorrect. datavar(::Type, [ dims... ]) requires `Type` as a first argument, but `$(type)` is not a `Type`."
    checktype = :(GraphPPL.ensure_type($(type)) || error($errstr))
    return :($checktype; $varexpr = ReactiveMP.datavar($model, $options, $(GraphPPL.fquote(varexpr)), ReactiveMP.PointMass{$type}, $(arguments...)); $varexpr)
end

function GraphPPL.write_constvar_expression(::RxInferBackend, model, varexpr, arguments)
    return :($varexpr = ReactiveMP.constvar($model, $(GraphPPL.fquote(varexpr)), $(arguments...)); $varexpr)
end

function GraphPPL.write_as_variable(::RxInferBackend, model, varexpr)
    return :(ReactiveMP.as_variable($model, $varexpr))
end

function GraphPPL.write_undo_as_variable(::RxInferBackend, varexpr)
    return :(ReactiveMP.undo_as_variable($varexpr))
end

function GraphPPL.write_anonymous_variable(::RxInferBackend, model, varexpr)
    return :(ReactiveMP.setanonymous!($varexpr, true))
end

function GraphPPL.write_make_node_expression(::RxInferBackend, model, fform, variables, options, nodeexpr, varexpr)
    return :($nodeexpr = ReactiveMP.make_node($model, $options, $fform, $varexpr, $(variables...)))
end

function GraphPPL.write_make_auto_node_expression(::RxInferBackend, model, rhs, nodeexpr, varexpr)
    return :($nodeexpr = ReactiveMP.make_node($model, RxInfer.AutoNode(), $varexpr, $rhs))
end

function GraphPPL.write_broadcasted_make_node_expression(::RxInferBackend, model, fform, variables, options, nodeexpr, varexpr)
    return :($nodeexpr = ReactiveMP.make_node.($model, $options, $fform, $varexpr, $(variables...)))
end

function GraphPPL.write_broadcasted_make_auto_node_expression(::RxInferBackend, model, rhs, nodeexpr, varexpr)
    return :($nodeexpr = ReactiveMP.make_node.($model, RxInfer.AutoNode(), $varexpr, $rhs))
end

function GraphPPL.write_autovar_make_node_expression(::RxInferBackend, model, fform, variables, options, nodeexpr, varexpr, autovarid)
    return :(($nodeexpr, $varexpr) = ReactiveMP.make_node($model, $options, $fform, RxInfer.AutoVar($(GraphPPL.fquote(autovarid))), $(variables...)))
end

function GraphPPL.write_autovar_make_auto_node_expression(::RxInferBackend, model, rhs, nodeexpr, varexpr, autovarid)
    return :(($nodeexpr, $varexpr) = ReactiveMP.make_node($model, RxInfer.AutoNode(), RxInfer.AutoVar($(GraphPPL.fquote(autovarid))), $rhs))
end

function GraphPPL.write_check_variable_existence(::RxInferBackend, model, varid, errormsg)
    return :(Base.haskey($model, $(QuoteNode(varid))) || Base.error($errormsg))
end

function GraphPPL.write_node_options(::RxInferBackend, model, fform, variables, options)
    is_factorisation_option_present = false
    is_meta_option_present          = false
    is_pipeline_option_present      = false

    factorisation_option = :(nothing)
    meta_option          = :(nothing)
    pipeline_option      = :(nothing)

    foreach(options) do option
        # Factorisation constraint option
        if @capture(option, q = fconstraint_)
            !is_factorisation_option_present || error("Factorisation constraint option $(option) for $(fform) has been redefined.")
            is_factorisation_option_present = true
            factorisation_option = write_fconstraint_option(fform, variables, fconstraint)
        elseif @capture(option, meta = fmeta_)
            !is_meta_option_present || error("Meta specification option $(option) for $(fform) has been redefined.")
            is_meta_option_present = true
            meta_option = write_meta_option(fform, fmeta)
        elseif @capture(option, pipeline = fpipeline_)
            !is_pipeline_option_present || error("Pipeline specification option $(option) for $(fform) has been redefined.")
            is_pipeline_option_present = true
            pipeline_option = write_pipeline_option(fform, fpipeline)
        else
            error("Unknown option '$option' for '$fform' node")
        end
    end

    return :(ReactiveMP.FactorNodeCreationOptions($factorisation_option, $meta_option, $pipeline_option))
end

# Meta helper functions

function write_meta_option(fform, fmeta)
    return :($fmeta)
end

# Pipeline helper functions

function write_pipeline_option(fform, fpipeline)
    if @capture(fpipeline, +(stages__))
        return :(+($(map(stage -> write_pipeline_stage(fform, stage), stages)...)))
    else
        return :($(write_pipeline_stage(fform, fpipeline)))
    end
end

function write_pipeline_stage(fform, stage)
    if @capture(stage, Default())
        return :(ReactiveMP.DefaultFunctionalDependencies())
    elseif @capture(stage, RequireEverything())
        return :(ReactiveMP.RequireEverythingFunctionalDependencies())
    elseif @capture(stage, (RequireInbound(args__)) | (RequireMessage(args__)) | (RequireMarginal(args__)))
        specs = map(args) do arg
            if @capture(arg, name_Symbol)
                return (name, :nothing)
            elseif @capture(arg, name_Symbol = dist_)
                return (name, dist)
            else
                error("Invalid arg specification in node's functional dependencies list: $(arg). Should be either `name` or `name = initial` expression")
            end
        end

        indices  = Expr(:tuple, map(s -> :(ReactiveMP.interface_get_index(Val{$(GraphPPL.fquote(fform))}, Val{$(GraphPPL.fquote(first(s)))})), specs)...)
        initials = Expr(:tuple, map(s -> :($(last(s))), specs)...)

        if @capture(stage, (RequireInbound(args__)) | (RequireMessage(args__)))
            return :(ReactiveMP.RequireMessageFunctionalDependencies($indices, $initials))
        elseif @capture(stage, RequireMarginal(args__))
            return :(ReactiveMP.RequireMarginalFunctionalDependencies($indices, $initials))
        else
            error("Unreacheable reached in `write_pipeline_stage`.")
        end
    else
        return stage
    end
end

# Factorisation constraint helper functions

function factorisation_replace_var_name(varnames, arg::Expr)
    index = findfirst(==(arg), varnames)
    return index === nothing ? error("Invalid factorisation argument: $arg. $arg should be available within tilde expression") : index
end

function factorisation_replace_var_name(varnames, arg::Symbol)
    index = findfirst(==(arg), varnames)
    return index === nothing ? arg : index
end

function factorisation_name_to_index(form, name)
    return :(ReactiveMP.interface_get_index(Val{$(GraphPPL.fquote(form))}, Val{ReactiveMP.interface_get_name(Val{$(GraphPPL.fquote(form))}, Val{$(GraphPPL.fquote(name))})}))
end

function check_uniqueness(t)
    return TupleTools.minimum(TupleTools.diff(TupleTools.sort(TupleTools.flatten(t)))) > 0
end

function sorted_factorisation(t)
    subfactorisations = map(TupleTools.sort, t)
    firstindices      = map(first, subfactorisations)
    staticlength      = TupleTools.StaticLength(length(firstindices))
    withindices       = ntuple(i -> (i, firstindices[i]), staticlength)
    permutation       = map(first, TupleTools.sort(withindices; by = last))
    return ntuple(i -> subfactorisations[permutation[i]], staticlength)
end

function write_fconstraint_option(form, variables, fconstraint)
    if @capture(fconstraint, (*(factors__)) | (q(names__)))
        factors = factors === nothing ? [fconstraint] : factors

        indexed = map(factors) do factor
            @capture(factor, q(names__)) || error("Invalid factorisation constraint: $factor")
            return map((n) -> RxInfer.factorisation_name_to_index(form, n), map((n) -> RxInfer.factorisation_replace_var_name(variables, n), names))
        end

        factorisation = Expr(:tuple, map(f -> Expr(:tuple, f...), indexed)...)
        errorstr = """Invalid factorisation constraint: ($fconstraint). Arguments are not unique, check node's interface names and model specification variable names."""

        return :(RxInfer.check_uniqueness($factorisation) ? RxInfer.sorted_factorisation($factorisation) : error($errorstr))
    elseif @capture(fconstraint, MeanField())
        return :(ReactiveMP.MeanField())
    elseif @capture(fconstraint, FullFactorisation())
        return :(ReactiveMP.FullFactorisation())
    else
        error("Invalid factorisation constraint: $fconstraint")
    end
end

## 

function GraphPPL.write_randomvar_options(::RxInferBackend, variable, options)
    is_pipeline_option_present                     = false
    is_prod_constraint_option_present              = false
    is_prod_strategy_option_present                = false
    is_marginal_form_constraint_option_present     = false
    is_marginal_form_check_strategy_option_present = false
    is_messages_form_constraint_option_present     = false
    is_messages_form_check_strategy_option_present = false

    pipeline_option                     = :(nothing)
    prod_constraint_option              = :(nothing)
    prod_strategy_option                = :(nothing)
    marginal_form_constraint_option     = :(nothing)
    marginal_form_check_strategy_option = :(nothing)
    messages_form_constraint_option     = :(nothing)
    messages_form_check_strategy_option = :(nothing)

    foreach(options) do option
        if @capture(option, pipeline = value_)
            !is_pipeline_option_present || error("`pipeline` option $(option) for random variable $(variable) has been redefined.")
            is_pipeline_option_present = true
            pipeline_option = value
        elseif @capture(option, $(:(prod_constraint)) = value_)
            !is_prod_constraint_option_present || error("`prod_constraint` option $(option) for random variable $(variable) has been redefined.")
            is_prod_constraint_option_present = true
            prod_constraint_option = value
        elseif @capture(option, $(:(prod_strategy)) = value_)
            !is_prod_strategy_option_present || error("`prod_strategy` option $(option) for random variable $(variable) has been redefined.")
            is_prod_strategy_option_present = true
            prod_strategy_option = value
        elseif @capture(option, $(:(marginal_form_constraint)) = value_)
            !is_marginal_form_constraint_option_present || error("`marginal_form_constraint` option $(option) for random variable $(variable) has been redefined.")
            is_marginal_form_constraint_option_present = true
            marginal_form_constraint_option = value
        elseif @capture(option, $(:(form_constraint)) = value_) # backward compatibility
            @warn "`form_constraint` option is deprecated. Use `marginal_form_constraint` option for variable $(variable) instead."
            !is_marginal_form_constraint_option_present || error("`marginal_form_constraint` option $(option) for random variable $(variable) has been redefined.")
            is_marginal_form_constraint_option_present = true
            marginal_form_constraint_option = value
        elseif @capture(option, $(:(marginal_form_check_strategy)) = value_)
            !is_marginal_form_check_strategy_option_present || error("`marginal_form_check_strategy` option $(option) for random variable $(variable) has been redefined.")
            is_marginal_form_check_strategy_option_present = true
            marginal_form_check_strategy_option = value
        elseif @capture(option, $(:(messages_form_constraint)) = value_)
            !is_messages_form_constraint_option_present || error("`messages_form_constraint` option $(option) for random variable $(variable) has been redefined.")
            is_messages_form_constraint_option_present = true
            messages_form_constraint_option = value
        elseif @capture(option, $(:(messages_form_check_strategy)) = value_)
            !is_messages_form_check_strategy_option_present || error("`messages_form_check_strategy` option $(option) for random variable $(variable) has been redefined.")
            is_messages_form_check_strategy_option_present = true
            messages_form_check_strategy_option = value
        else
            error("Unknown option '$option' for randomv variable '$variable'.")
        end
    end

    return :(ReactiveMP.RandomVariableCreationOptions(
        $pipeline_option,
        nothing, # it does not make a lot of sense to override `proxy_variables` option
        $prod_constraint_option,
        $prod_strategy_option,
        $marginal_form_constraint_option,
        $marginal_form_check_strategy_option,
        $messages_form_constraint_option,
        $messages_form_check_strategy_option
    ))
end

function GraphPPL.write_datavar_options(::RxInferBackend, variable, type, options)
    is_subject_option_present       = false
    is_allow_missing_option_present = false

    # default options
    subject_option       = :(nothing)
    allow_missing_option = :(Val(false))

    foreach(options) do option
        if @capture(option, subject = value_)
            !is_subject_option_present || error("`subject` option $(option) for data variable $(variable) has been redefined.")
            is_subject_option_present = true
            subject_option = value
        elseif @capture(option, $(:(allow_missing)) = value_)
            !is_allow_missing_option_present || error("`allow_missing` option $(option) for data variable $(variable) has been redefined.")
            is_allow_missing_option_present = true
            allow_missing_option = :(Val($value))
        else
            error("Unknown option '$option' for data variable '$variable'.")
        end
    end

    return :(ReactiveMP.DataVariableCreationOptions(ReactiveMP.PointMass{$type}, $subject_option, $allow_missing_option))
end

# Constraints specification language

## Factorisations constraints specification language

function GraphPPL.write_constraints_specification(::RxInferBackend, factorisation, marginalsform, messagesform, options)
    return :(ReactiveMP.ConstraintsSpecification($factorisation, $marginalsform, $messagesform, $options))
end

function GraphPPL.write_constraints_specification_options(::RxInferBackend, options)
    @capture(options, [entries__]) || error("Invalid constraints specification options syntax. Should be `@constraints [ option1 = value1, ... ] ...`, but `$(options)` found.")

    is_warn_option_present = false

    warn_option = :(true)

    foreach(entries) do option
        if @capture(option, warn = value_)
            !is_warn_option_present || error("`warn` option $(option) for constraints specification has been redefined.")
            is_warn_option_present = true
            @assert value isa Bool "`warn` option for constraints specification expects true/false value"
            warn_option = value
        else
            error("Unknown option '$option' for constraints specification.")
        end
    end

    return :(ReactiveMP.ConstraintsSpecificationOptions($warn_option))
end

function GraphPPL.write_factorisation_constraint(::RxInferBackend, names, entries)
    return :(ReactiveMP.FactorisationConstraintsSpecification($names, $entries))
end

function GraphPPL.write_factorisation_constraint_entry(::RxInferBackend, names, entries)
    return :(ReactiveMP.FactorisationConstraintsEntry($names, $entries))
end

function GraphPPL.write_init_factorisation_not_defined(::RxInferBackend, spec, name)
    return :($spec = ReactiveMP.FactorisationSpecificationNotDefinedYet{$(QuoteNode(name))}())
end

function GraphPPL.write_check_factorisation_is_not_defined(::RxInferBackend, spec)
    return :($spec isa ReactiveMP.FactorisationSpecificationNotDefinedYet)
end

function GraphPPL.write_factorisation_split(::RxInferBackend, left, right)
    return :(ReactiveMP.factorisation_split($left, $right))
end

function GraphPPL.write_factorisation_combined_range(::RxInferBackend, left, right)
    return :(ReactiveMP.CombinedRange($left, $right))
end

function GraphPPL.write_factorisation_splitted_range(::RxInferBackend, left, right)
    return :(ReactiveMP.SplittedRange($left, $right))
end

function GraphPPL.write_factorisation_functional_index(::RxInferBackend, repr, fn)
    return :(ReactiveMP.FunctionalIndex{$(QuoteNode(repr))}($fn))
end

function GraphPPL.write_form_constraint_specification_entry(::RxInferBackend, T, args, kwargs)
    return :(ReactiveMP.make_form_constraint($T, $args...; $kwargs...))
end

function GraphPPL.write_form_constraint_specification(::RxInferBackend, specification)
    return :(ReactiveMP.FormConstraintSpecification($specification))
end

## Meta specification language

function GraphPPL.write_meta_specification(::RxInferBackend, entries, options)
    return :(ReactiveMP.MetaSpecification($entries, $options))
end

function GraphPPL.write_meta_specification_options(::RxInferBackend, options)
    @capture(options, [entries__]) || error("Invalid meta specification options syntax. Should be `@meta [ option1 = value1, ... ] ...`, but `$(options)` found.")

    is_warn_option_present = false

    warn_option = :(true)

    foreach(entries) do option
        if @capture(option, warn = value_)
            !is_warn_option_present || error("`warn` option $(option) for meta specification has been redefined.")
            is_warn_option_present = true
            @assert value isa Bool "`warn` option for meta specification expects true/false value"
            warn_option = value
        else
            error("Unknown option '$option' for meta specification.")
        end
    end

    return :(ReactiveMP.MetaSpecificationOptions($warn_option))
end

function GraphPPL.write_meta_specification_entry(::RxInferBackend, F, N, meta)
    return :(ReactiveMP.MetaSpecificationEntry(Val($F), Val($N), $meta))
end

# Aliases

ReactiveMPNodeAliases = (
    (
        (expression) -> @capture(expression, a_ || b_) ? :(ReactiveMP.OR($a, $b)) : expression,
        "`a || b`: alias for `OR(a, b)` node (operator precedence between `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, a_ && b_) ? :(ReactiveMP.AND($a, $b)) : expression,
        "`a && b`: alias for `AND(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, a_ -> b_) ? :(ReactiveMP.IMPLY($a, $b)) : expression,
        "`a -> b`: alias for `IMPLY(a, b)` node (operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    (
        (expression) -> @capture(expression, (¬a_) | (!a_)) ? :(ReactiveMP.NOT($a)) : expression,
        "`¬a` and `!a`: alias for `NOT(a)` node (Unicode `\\neg`, operator precedence `||`, `&&`, `->` and `!` is the same as in Julia)."
    ),
    ((expression) -> @capture(expression, +(args__)) ? GraphPPL.fold_linear_operator_call(expression) : expression, "`a + b + c`: alias for `(a + b) + c`"),
    ((expression) -> @capture(expression, *(args__)) ? GraphPPL.fold_linear_operator_call(expression) : expression, "`a * b * c`: alias for `(a * b) * c`"),
    (
        (expression) -> if @capture(expression, (Normal | Gaussian)((μ) | (m) | (mean) = mean_, (σ²) | (τ⁻¹) | (v) | (var) | (variance) = var_))
            :(NormalMeanVariance($mean, $var))
        else
            expression
        end,
        "`Normal(μ|m|mean = ..., σ²|τ⁻¹|v|var|variance = ...)` alias for `NormalMeanVariance(..., ...)` node. `Gaussian` could be used instead `Normal` too."
    ),
    (
        (expression) -> if @capture(expression, (Normal | Gaussian)((μ) | (m) | (mean) = mean_, (τ) | (γ) | (σ⁻²) | (w) | (p) | (prec) | (precision) = prec_))
            :(NormalMeanPrecision($mean, $prec))
        else
            expression
        end,
        "`Normal(μ|m|mean = ..., τ|γ|σ⁻²|w|p|prec|precision = ...)` alias for `NormalMeanVariance(..., ...)` node. `Gaussian` could be used instead `Normal` too."
    ),
    (
        (expression) -> if @capture(expression, (MvNormal | MvGaussian)((μ) | (m) | (mean) = mean_, (Σ) | (V) | (Λ⁻¹) | (cov) | (covariance) = cov_))
            :(MvNormalMeanCovariance($mean, $cov))
        else
            expression
        end,
        "`MvNormal(μ|m|mean = ..., Σ|V|Λ⁻¹|cov|covariance = ...)` alias for `MvNormalMeanCovariance(..., ...)` node. `MvGaussian` could be used instead `MvNormal` too."
    ),
    (
        (expression) -> if @capture(expression, (MvNormal | MvGaussian)((μ) | (m) | (mean) = mean_, (Λ) | (W) | (Σ⁻¹) | (prec) | (precision) = prec_))
            :(MvNormalMeanPrecision($mean, $prec))
        else
            expression
        end,
        "`MvNormal(μ|m|mean = ..., Λ|W|Σ⁻¹|prec|precision = ...)` alias for `MvNormalMeanPrecision(..., ...)` node. `MvGaussian` could be used instead `MvNormal` too."
    ),
    (
        (expression) -> if @capture(expression, (MvNormal | MvGaussian)((μ) | (m) | (mean) = mean_, (τ) | (γ) | (σ⁻²) | (scale_diag_prec) | (scale_diag_precision) = scale_))
            :(MvNormalMeanScalePrecision($mean, $scale))
        else
            expression
        end,
        "`MvNormal(μ|m|mean = ..., τ|γ|σ⁻²|scale_diag_prec|scale_diag_precision = ...)` alias for `MvNormalMeanScalePrecision(..., ...)` node. `MvGaussian` could be used instead `MvNormal` too."
    ),
    (
        (expression) -> if @capture(expression, (Normal | Gaussian)(args__))
            error(
                "Please use a specific version of the `Normal` (`Gaussian`) distribution (e.g. `NormalMeanVariance` or aliased version `Normal(mean = ..., variance|precision = ...)`)."
            )
        else
            expression
        end,
        missing
    ),
    (
        (expression) -> if @capture(expression, (MvNormal | MvGaussian)(args__))
            error(
                "Please use a specific version of the `MvNormal` (`MvGaussian`) distribution (e.g. `MvNormalMeanCovariance` or aliased version `MvNormal(mean = ..., covariance|precision = ...)`)."
            )
        else
            expression
        end,
        missing
    ),
    (
        (expression) -> @capture(expression, Gamma((α) | (a) | (shape) = shape_, (θ) | (β⁻¹) | (scale) = scale_)) ? :(GammaShapeScale($shape, $scale)) : expression,
        "`Gamma(α|a|shape = ..., θ|β⁻¹|scale = ...)` alias for `GammaShapeScale(..., ...) node.`"
    ),
    (
        (expression) -> @capture(expression, Gamma((α) | (a) | (shape) = shape_, (β) | (θ⁻¹) | (rate) = rate_)) ? :(GammaShapeRate($shape, $rate)) : expression,
        "`Gamma(α|a|shape = ..., β|θ⁻¹|rate = ...)` alias for `GammaShapeRate(..., ...) node.`"
    )
)

function GraphPPL.show_tilderhs_alias(::RxInferBackend, io = stdout)
    foreach(skipmissing(map(last, ReactiveMPNodeAliases))) do alias
        println(io, "- ", alias)
    end
end

function apply_alias_transformation(notanexpression, alias)
    # We always short-circuit on non-expression
    return (notanexpression, true)
end

function apply_alias_transformation(expression::Expr, alias)
    _expression = first(alias)(expression)
    # Returns potentially modified expression and a Boolean flag, 
    # which indicates if expression actually has been modified
    return (_expression, _expression !== expression)
end

function GraphPPL.write_inject_tilderhs_aliases(::RxInferBackend, model, tilderhs)
    return postwalk(tilderhs) do expression
        # We short-circuit if `mflag` is true
        _expression, _ = foldl(ReactiveMPNodeAliases; init = (expression, false)) do (expression, mflag), alias
            return mflag ? (expression, true) : apply_alias_transformation(expression, alias)
        end
        return _expression
    end
end

##

"""

```julia
@model function model_name(model_arguments...; model_keyword_arguments...)
    # model description
end
```

`@model` macro generates a function that returns an equivalent graph-representation of the given probabilistic model description.

## Supported alias in the model specification
$(begin io = IOBuffer(); GraphPPL.show_tilderhs_alias(RxInferBackend(), io); String(take!(io)) end)
"""
macro model end

macro model(model_specification)
    return GraphPPL.generate_model_expression(RxInferBackend(), model_specification)
end

macro constraints(constraints_specification)
    return GraphPPL.generate_constraints_expression(RxInferBackend(), :([]), constraints_specification)
end

macro constraints(constraints_options, constraints_specification)
    return GraphPPL.generate_constraints_expression(RxInferBackend(), constraints_options, constraints_specification)
end

macro meta(meta_specification)
    return GraphPPL.generate_meta_expression(RxInferBackend(), :([]), meta_specification)
end

macro meta(meta_options, meta_specification)
    return GraphPPL.generate_meta_expression(RxInferBackend(), meta_options, meta_specification)
end
