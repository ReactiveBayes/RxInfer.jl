module OptimExt

# The default optimiser of a continuous univariate point mass, `PointMassFormConstraint()`, which
# RxInfer runs through Optim.jl when it is loaded.

using RxInfer, Optim
using RxInfer:
    OptimBackend, PointMassFormConstraint, call_boundaries, call_starting_point

function RxInfer.point_mass_optimize(
        ::OptimBackend, constraint::PointMassFormConstraint, distribution
    )
    target = let distribution = distribution
        (x) -> -logpdf(distribution, x[1])
    end

    lower, upper = call_boundaries(constraint, distribution)

    result = if isinf(lower) && isinf(upper)
        Optim.optimize(
            target,
            call_starting_point(constraint, distribution),
            Optim.LBFGS(),
        )
    else
        Optim.optimize(
            target,
            [lower],
            [upper],
            call_starting_point(constraint, distribution),
            Optim.Fminbox(Optim.GradientDescent()),
        )
    end

    if Optim.converged(result)
        return PointMass(Optim.minimizer(result)[1])
    else
        error(
            "Optimisation procedure for point mass estimation did not converge",
            result,
        )
    end
end

end
