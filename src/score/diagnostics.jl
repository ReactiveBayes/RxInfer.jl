"""
    apply_diagnostic_check(check, stream)

This function applies a `check` to the `stream`. Does nothing if `check` is of type `Nothing`. 
"""
apply_diagnostic_check(::Nothing, stream) = stream

"""
    ObjectiveDiagnosticCheckNaNs

If enabled checks that both variable and factor bound score functions in the objective computation do not return `NaN`s. 
Throws an error if finds `NaN`. 
"""
struct ObjectiveDiagnosticCheckNaNs end

check_isnan(something)              = isnan(something)
check_isnan(counting::CountingReal) = check_isnan(BayesBase.value(counting))

function apply_diagnostic_check(::ObjectiveDiagnosticCheckNaNs, stream)
    error_fn = (_) -> """
        Failed to compute the final objective value. The result is `NaN`.
        Use `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
    """
    return stream |> error_if(check_isnan, error_fn)
end

"""
    ObjectiveDiagnosticCheckInfs

If enabled checks that both variable and factor bound score functions in the objective computation do not return `Inf`s. 
Throws an error if finds `Inf`. 
"""
struct ObjectiveDiagnosticCheckInfs end

check_isinf(something)              = isinf(something)
check_isinf(counting::CountingReal) = check_isinf(BayesBase.value(counting))

function apply_diagnostic_check(::ObjectiveDiagnosticCheckInfs, stream)
    error_fn = (_) -> """
        Failed to compute the final objective value. The result is `Inf`.
        Use `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
    """
    return stream |> error_if(check_isinf, error_fn)
end

apply_diagnostic_check(::Nothing, something, stream)     = stream
apply_diagnostic_check(checks::Tuple, something, stream) = foldl((folded, check) -> apply_diagnostic_check(check, something, folded), checks; init = stream)

const DefaultObjectiveDiagnosticChecks = (ObjectiveDiagnosticCheckNaNs(), ObjectiveDiagnosticCheckInfs())