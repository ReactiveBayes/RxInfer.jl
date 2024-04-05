# [Bethe Free Energy implementation in RxInfer](@id lib-bethe-free-energy)

The following text introduces the Bethe Free Energy. We start be defining a factorized model and move from the Variational Free Energy to a definition of the Bethe Free Energy.

## [Factorized model](@id lib-bethe-factorized-model)

Before we can define a model, we must identify all variables that are relevant to the problem at hand. We distinguish between variables that can be directly observed,
$$y = (y_1, \dots, y_j, \dots, y_m)\,,$$
and variables that can not be observed directly, also known as latent variables,
$$x = (x_1, \dots, x_i, \dots, x_n)\,.$$
We then define a model that factorizes over consituent smaller factors (functions), as
$$f(y,x) = \prod_a f_a(y_a,x_a)\,.$$

Individual factors may represent stochastic functions, such as conditional or prior distributions, but also potential functions or deterministic relationships. A factor may depend on multiple observed and/or latent variables (or none).


## [Variational Free Energy](@id lib-bethe-vfe)

The Variational Free Energy (VFE) then defines a functional objective that includes the model and a variational distribution over the latent variables,
$$F[q](\hat{y}) = \mathbb{E}_{q(x)}\left[\log \frac{q(x)}{f(y=\hat{y}, x)} \right]\,.$$
A functional defines a function of a function that returns a scalar. Here, the VFE is a function of the variational distribution (as indicated by square brackets) and returns a number.

The VFE is also a function of the observed data, as indicated by round brackets, where the data are substituted in the factorized model.

## [Variational inference](@id lib-bethe-variational-inference)

The goal of variational inference is to find a variational distibution that minimizes the VFE,
$$q^{*}(x) = \arg\min_{q\in\mathcal{Q}} F[q](\hat{y})\,.$$
This objective can be optimized (under specific constraints) with the use of variational calculus. Constraints are implied by the domain over which the variational distribution is optimized, and can be enforced by Lagrange multipliers.

For the VFE, constraints enforce e.g. the normalization of the variational distribution. The variational distribution that minimizes the VFE then approximates the true (but often unobtainable) posterior distribution.

## [Bethe approximation](@id lib-bethe-approximation)
Optimization of the VFE is still a daunting task, because the variational distribution is a joint distribution over possibly many latent variables. Instead of optimizing the joint variational distribution directly, a factorized variational distribution is often chosen. The factorized variational distribution is then optimized for its constituent factors.

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

For iterative optimization of the BFE, the variational distributions must first be initialized. The `initmarginals` keyword argument to the [`infer`](@ref) function initializes the variational distributions of the BFE.

For disambiguation, note that the initialization of the variational distribution is a different design consideration than the choice of priors. A prior specifies a factor in the model definition, while initialization concerns factors in the variational distribution.


## [Further reading](@id lib-bethe-further-reading)

- [Pearl (1986)](https://dl.acm.org/doi/pdf/10.1145/3501714.3501728) on the original foundations of Bayesian networks and belief propagation;
- [Yedidia et al. (2005)](https://ieeexplore.ieee.org/iel5/18/31406/01459044.pdf) on the connections between belief propagation and regional approximations to the VFE;
- [Dauwels (2007)](https://ieeexplore.ieee.org/iel5/4497218/4557062/04557602.pdf) on variational message passing on Forney-style factor graphs (FFGs);
- [Senoz et al. (2021)](https://www.mdpi.com/1099-4300/23/7/807/htm) on constraint manipulation and message passing on FFGs.

## Implementation details 

`RxInfer` implements Bethe Free Energy optimization in an implicit way via the mesasge passing technique. That means that the inference engine does not compute BFE values explicitly, 
unless specified explicitly. The [`infer`](@ref) function has `free_energy` flag, which indicates whether BFE values must be computed explicitly or not. Note, however, that due to the reactive nature of the message passing implementation in `RxInfer` the computed BFE value may not represent its actual state. This may happen when updates for certain posteriors arriving more often than updates for other posteriors and usually tend to happen in models with loops in its structure. To circumvent this, instead of checking if BFE value is being minimized it is advised to check if it __converges__.

```@docs
RxInfer.BetheFreeEnergy
RxInfer.BetheFreeEnergyDefaultMarginalSkipStrategy
RxInfer.BetheFreeEnergyDefaultScheduler
RxInfer.ReactiveMPFreeEnergyPlugin
```

### Extra diagnostic checks

```@docs
RxInfer.apply_diagnostic_check
RxInfer.ObjectiveDiagnosticCheckNaNs
RxInfer.ObjectiveDiagnosticCheckInfs
RxInfer.DefaultObjectiveDiagnosticChecks
```
