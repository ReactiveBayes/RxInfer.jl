# [Bethe Free Energy implementation in RxInfer](@id lib-bethe-free-energy)

The following text aims to introduce the Bethe Free Energy with as little mathematics as needed (but no less). We start be defining a factorized model and move from the Variational Free Energy to a definition for the Bethe Free Energy.

## [Factorized model](@id lib-bethe-factorized-model)

Before we can define a model, we must identify all variables that are relevant to the problem at hand. We distinguish between variables that can be directly observed,
$$y = (y_1, \dots, y_j, \dots, y_m)\,,$$
and variables that can not be observed (latent variables),
$$x = (x_1, \dots, x_i, \dots, x_n)\,.$$
We then define a model that factorizes over consituent smaller factors (functions), as
$$f(y,x) = \prod_a f_a(y_a,x_a)\,.$$

Individual factors may represent stochastic functions, such as conditional or prior distributions, but also potential functions or deterministic relationships. A factor may depend on multiple observed and/or latent variables (or none).


## [Variational Free Energy](@id lib-bethe-vfe)

The Variational Free Energy (VFE) defines a functional objective that includes the model and a variational distribution over the latent variables,
$$F[q](\hat{y}) = \mathbb{E}_{q(x)}\left[\log \frac{q(x)}{f(y=\hat{y}, x)} \right]\,.$$
A functional defines a function of a function that returns a scalar. The VFE is a function of the variational distribution (as indicated by square brackets) and returns a number.

The VFE is also a function of the observed data, as indicated by round brackets. The data are substituted in the factorized model.


## [Variational inference](@id lib-bethe-variational-inference)

The goal of variational inference is to find a variational distibution that minimizes the VFE,
$$q^{*}(x) = \arg\min_{q\in\mathcal{Q}} F[q](\hat{y})\,.$$
This objective can be optimized with variational calculus, under specific constraints. 

For the VFE, constraints enforce e.g. the normalization of the variational distribution. The variational distribution that minimizes the VFE then approximates the true (but often unobtainable) posterior distribution.

## [Bethe approximation](@id lib-bethe-approximation)
Optimization of the VFE is still a daunting task, because the variational distribution is a joint distribution over possibly many latent variables. Instead of optimizing the joint variational distribution directly, a factorized variational distribution is often chosen, and optimized for its constituent factors.

A popular choice of factorization is the Bethe approximation, which is constructed from the factorization of the model itself,
$$q(x) \triangleq \frac{\prod_a q_a(x_a)}{\prod_i q_i(x_i)^{d_i - 1}}\,.$$
The numerator iterates over the factors in the model, and carves the joint variational distribution in smaller variational distributions that are more manageable to optimize.

The denominator of the Bethe approximation iterates over all individual latent variables and discounts them. The discounting factor is chosen as the degree of the variable minus one, where the degree counts the number of factors in which the variable appears.

The Bethe approximation thus constrains the variational distribution to a factorized form. However, the true posterior distribution might not factorize in this way, e.g. if the grapical representation of the model contains cycles. In these cases the Bethe approximation trades the exact solution for computational tractability.


## [Bethe Free Energy](@id lib-bethe-bfe)

The Bethe Free Energy (BFE) substitutes the Bethe approximation in the VFE, which then fragments over factors and variables, as
$$F_B[q](\hat{y}) = \sum_a U_a[q_a](\hat{y}_a) - \sum_a H[q_a] + \sum_i (d_i - 1) H[q_i]\,.$$
The first term of the BFE specifies an average energy, 
$$U_a[q_a](\hat{y}_a) = -\mathbb{E}_{q_a(x_a)}\left[\log f_a(y_a=\hat{y}_a, x_a)\right]\,,$$
which internalizes the factors of the  model. The last two terms specify entropies.

Crucially, the BFE can be iteratively optimized for each individual variational distribution in turn. Optimization of the BFE is thus more manageable than direct optimization of the VFE.

For iterative optimization of the BFE, the variational distributions must first be initialized. The `initmarginals` keyword argument to the [`inference`](@ref) and [`rxinference`](@ref) functions initializes the variational distributions of the BFE.


```@docs
RxInfer.AbstractScoreObjective
RxInfer.BetheFreeEnergy
RxInfer.apply_diagnostic_check
RxInfer.BetheFreeEnergyCheckNaNs
RxInfer.BetheFreeEnergyCheckInfs
```
