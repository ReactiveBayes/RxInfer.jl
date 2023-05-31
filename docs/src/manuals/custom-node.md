
# Creating your own custom nodes


Welcome to the `RxInfer` documentation on creating custom factor graph nodes. In `RxInfer`, factor nodes represent functional relationships between variables, also known as factors. Together, these factors define your probabilistic model. Quite often these factors represent distributions, denoting how a certain parameter affects another. However, other factors are also possible, such as ones specifying linear or non-linear relationships. `RxInfer` already supports a lot of factor nodes, however, depending on the problem that you are trying to solve, you may need to create a custom node that better fits the specific requirements of your model. This tutorial will guide you through the process of defining a custom node in `RxInfer`, step by step. By the end of this tutorial, you will be able to create your own custom node and integrate it into your model.


To create a custom node in `RxInfer`, 4 steps are required:

1. Create your custom node in `RxInfer` using the `@node` macro.
2. Define the corresponding message passing update rules with the `@rule` macro. These rules specify how the node processes information in the form of messages, and how it communicates the results to adjacent parts of the model.
3. Specify computations for marginal distributions of the relevant variables with the `@marginalrule` macro.
4. Implement the computation of the Free Energy in a node with the `@average_energy` macro.


Throughout this tutorial, we will create a node for the `Bernoulli` distribution. The `Bernoulli` distribution is a commonly used distribution in statistical modeling that is often used to model a binary outcome, such as a coin flip. By recreating this node, we will be able to demonstrate the process of creating a custom node, from notifying `RxInfer` of the nodes existence to implementing the required methods. While this tutorial focuses on the `Bernoulli` distribution, the principles can be applied to creating custom nodes for other distributions as well. So let's get started!


## Problem statement


Jane wants to determine whether a coin is a fair coin, meaning that is equally likely to land on heads or tails. In order to determine this, she will throw the coin $K=20$ times and write down how often it lands on heads and tails. The result of this experiment is a realization of the underlying stochastic process. Jane models the outcome of the experiment $x_k\in\{0,1\}$ using the Bernoulli distribution as

$$p(x_k \mid \pi) = \mathrm{Ber}(x_k \mid \pi) = \pi^{x_k} (1-\pi)^{1-x_k},$$

where $\pi \in[0,1]$ denotes the probability that she throws heads, also known as the success probability. Jane also has a prior belief (initial guess) about the value of $\pi$ which she models using the Beta distribution as

$$p(\pi) = \mathrm{Beta}(\pi \mid 4, 8).$$

With this prior belief, the total probabilistic model that she has for this experiment is given by

$$p(x_{1:K}, \pi) = p(\pi) \prod_{k=1}^K p(x_k \mid \pi).$$

Jane is interested in determining the fairness of the coin. Therefore she aims to infer (calculate) the posterior belief of $\pi$, $p(\pi \mid x_{1:K})$, denoting how $\pi$ is distributed after we have seen the data.


---


## Step 1: Creating the custom node


!!! note
    In this example we will assume that the `Bernoulli` node and distribution do not yet exist. The `RxInfer` already defines the node for the `Bernoulli` distribution from the `Distributions.jl` package.


First things first, let's import `RxInfer`:

```@example create-node
using RxInfer
```



In order to define a custom node using the `@node` macro from `RxInfer`, we need the following three arguments:
1. The name of the node. (`::Type`)
2. Whether the node is `Deterministic` or `Stochastic`.
3. The interfaces of the node and any potential aliases. (`::Vector`)


For the name of the node we wish to use `MyBernoulli` in this tutorial (`Bernoulli` already exists). However, the corresponding distribution does not yet exist. Therefore we need to specify it first as

```@example create-node
# struct for Bernoulli distribution with success probability π
struct MyBernoulli{T <: Real} <: ContinuousUnivariateDistribution
    π :: T
    new(π :: Real) = 0 ≤ π ≤ 1 ? MyBernoulli(π) : throw(ArgumentError("π must be between 0 and 1"))	
end

# for simplicity, let's also specify the mean of the distribution
Distributions.mean(d::MyBernoulli) = d.π

nothing # hide
```

In this case the distribution also has an entry in the struct, however, this is not necessary as long as the name of the distribution is a `Type`. The custom node created with `struct NewNode end` would also work fine.

!!! note 
    You can use regular functions, e.g `+` as a node type. Their Julia type, however, is written with the `typeof(_)` specification, e.g. `typeof(+)`

For our node we are dealing with a stochastic node, because the node forms a probabilistic relationship. This means that for a given value of $\pi$, we do know the corresponding value of the output, but we do have some belief about this. Deterministic nodes include for example linear and non-linear transformation.

The interfaces specify what variables are connected to the node. The first argument is its output by convention. The ordering is important for both the model specification as the rule definition. As an example consider the `NormalMeanVariance` factor node. This factor node has interfaces `[out, μ, v]` and can be called in the model specification language as `x ~ NormalMeanVariance(μ, v)`. It is also possible to use aliases for the interfaces, which can be specified in a tuple as you will see below.

Concluding, we can create the `MyBernoulli` factor node as

```@example create-node
@node MyBernoulli Stochastic [out, (π, aliases = [p])]
```

Cool! Step 1 is done, we have created a custom node.


---


## Step 2: Defining rules for our node


In order for `RxInfer` to perform probabilistic inference and compute posterior distributions, such as $p(\pi\mid x_{1:K})$, we need to tell it how to perform inference locally around our node. This localization is what makes `RxInfer` achieve high performance. In our message passing-based paradigm, we need to describe how the node processes incoming information in the form of messages (or marginals). Here we will highlight two different message passing strategies: sum-product message passing and variational message passing.


### Sum-product message passing update rules


In sum-product message passing we compute outgoing messages to our node as

$$\vec{\mu}(x) \propto \int \mathrm{Ber}(x\mid \pi) \vec{\mu}(\pi) \mathrm{d}x$$

$$\overleftarrow{\mu}(\pi) \propto \sum_{x \in \{0,1\}} \mathrm{Ber}(x\mid \pi) \overleftarrow{\mu}(x)$$

This integral does not always have nice tractable solutions. However, for some forms of the incoming messages, it does yield a tractable solution.



For the case of a `Beta` message coming into our node, the outgoing message will be the predictive posterior of the `Bernoulli` distribution with a `Beta` prior. Here we obtain $\pi = \frac{\alpha}{\alpha + \beta}$, which coincides with the mean of the `Beta` distribution. Hence, we can write down the first update rule using the `@rule` macro as

```@example create-node
@rule MyBernoulli(:out, Marginalisation) (m_π :: Beta,) = MyBernoulli(mean(m_π))
```



Here, `:out` refers to the interface of the outgoing message. The second argument denotes the incoming messages (which can be typed) as a tuple. Therefore make sure that it has a trailing `,` when there is a single message coming in. `m_π` is shorthand for "the incoming message on interface `π`". As we will see later, the structured approximation update rule for incoming message from `π` will have `q_π` as parameter.

The second rule is also straightforward; if `π` is a `PointMass` and therefore fixed, the outgoing message will be `MyBernoulli(π)`:

```@example create-node
@rule MyBernoulli(:out, Marginalisation) (m_π :: PointMass,) = MyBernoulli(mean(m_π))
```



Continuing with the sum-product update rules, we now have to define the update rules towards the `π` interface. We can only do exact inference if the incoming message is known, which in the case of the `Bernoulli` distribution, means that the `out` message is a `PointMass` distribution that is either `0` or `1`. The updated Beta distribution for `π` will be:

$$\overleftarrow{\mu}(π) \propto \mathrm{Beta}(1 + x, 2 - x)$$

Which gives us the following update rule:

```@example create-node
@rule MyBernoulli(:π, Marginalisation) (m_out :: PointMass,) = begin
    p = mean(m_out)
    Beta(one(p) + p, 2one(p) - p)
end
```



### Variational message passing update rules


We will now cover our second set of update rules. The sum-product messages are not always tractable and therefore we may need to resort to approximations. Here we highlight the variational approximation. In variational message passing we compute outgoing messages to our node as

$$\vec{\nu}(x) \propto \exp \int q(\pi) \ln \mathrm{Ber}(x\mid \pi) \mathrm{d}x$$

$$\overleftarrow{\nu}(\pi) \propto \exp \sum_{x \in \{0,1\}} q(x) \ln \mathrm{Ber}(x\mid \pi)$$

These messages depend on the marginals on the adjacent edges and not on the incoming messages as was the case with sum-product message passing. Update rules that operate on the marginals instead of the incoming messages are specified with the `q_{interface}` argument names. With these update rules, we can often support a wider family of distributions. Below we directly give the variational update rules. Deriving them yourself will be a nice challenge.

```@example create-node
#rules towards out
@rule MyBernoulli(:out, Marginalisation) (q_π :: PointMass,) = MyBernoulli(mean(q_π))

@rule Bernoulli(:out, Marginalisation) (q_π::Any,) = begin
    rho_1 = mean(log, q_π)          # E[ln(x)]
    rho_2 = mean(mirrorlog, q_π)    # E[log(1-x)]
    m = max(rho_1, rho_2)
    tmp = exp(rho_1 - m)
    p = clamp(tmp / (tmp + exp(rho_2 - m)), tiny, one(m))
    return Bernoulli(p)
end

#rules towards π
@rule MyBernoulli(:π, Marginalisation) (q_out :: Any,) = begin
    p = mean(q_out)
    return Beta(one(p) + p, 2one(p) - p)
end
```

!!! node
    Typically, the type of the variational distributions `q_` does not matter in the real computations, but only their statistics, e.g `mean` or `var`. Thus, in this case, we may safely use `::Any`.

In the example that we will show later on, we solely use sum-product message passing. Variational message passing requires us to set the local constraints in our model, something which is out of scope of this tutorial.


---


## Step 3: Defining joint marginals for our node


The entire probabilistic model can be scored using the Bethe free energy, which bounds the log-evidence for acyclic graphs. This Bethe free energy consists out of the sum of node-local entropies, negative node-local average energies and edge specific entropies. Formally we can denote this by

$$F[q,f] = - \sum_{a\in\mathcal{V}} \mathrm{H}[q_a(s_a)] - \sum_{a\in\mathcal{V}}\mathrm{E}_{q_a(s_a)}[\ln f_a(s_a)] + \sum_{i\in\mathcal{E}}\mathrm{H}[q_i(s_i)]$$

Here we call $q_a(s_a)$ the joint marginals around a node and $-\mathrm{E}_{q_a(s_a)}[\ln f_a(s_a)]$ we term the average energy.

In order to be able to compute the Bethe free energy, we need to first describe how to compute $q_a(s_a)$, defined in our case as 

$$q(x_k, \pi) = \vec{\mu}(\pi) \overleftarrow{\mu}(x_k) \mathrm{Ber}(x_k \mid \pi)$$

To calculate the updated posterior marginal for our custom distribution, we need to return joint posterior marginals for the interfaces of our node. In our case, the posterior marginal for the observation is still the same `PointMass` distribution. However, to calculate the posterior marginal over `π`, we use `RxInfer`'s built-in `prod` functionality to multiply the `Beta` prior with the `Beta` likelihood. This gives us the updated posterior distribution, which is also a `Beta` distribution. We use `ProdAnalytical()` parameter to ensure that we multiply the two distributions analytically. This is done as follows:

```@example create-node
@marginalrule MyBernoulli(:out_π) (m_out::PointMass, m_π::Beta) = begin
    r = mean(m_out)
    p = prod(ProdAnalytical(), Beta(one(r) + r, 2one(r) - r), m_π)
    return (out = m_out, p = p)
end
```

In this code `:out_π` describes the arguments of the joint marginal distribution. The second argument contains the incoming messages. Here we know from the model specification that we observe `out` and therefore this has to be a `PointMass`. Because it is a `PointMass`, the joint marginal automatically factorizes as $q(x_k, \pi) = q(x_k)q(\pi)$. These are the distributions that we return in a form of the `NamedTuple`. `NamedTuple` is used only in cases where we know that the joint marginal factorizes further, but typically it should be a full distribution. For computing $q(\pi)$ we need to compute the product $\vec{\mu}(\pi)\overleftarrow{\mu}(\pi)$. We already know how $\overleftarrow{\mu}(\pi)$ looks like from the previous step, so we can just use the `prod` function.


---


## Step 4: Defining the average energy for our node


To complete the computation of the Bethe free energy, we also need to compute the average energy term. The average energy in our `MyBernoulli` example can be computed as $-\mathrm{E}_{q(x_k, \pi)}[\ln p(x_k \mid \pi)]$, however, because we know that we observe $x_k$ and therefore $q(x_k, \pi)$ factorizes, we can instead compute
$$\begin{aligned}
-\mathrm{E}_{q(x_k)q(\pi)}[\ln p(x_k \mid \pi)]
&= -\mathrm{E}_{q(x_k)q(\pi)} [\ln (\pi^{x_k} (1-\pi)^{1 - x_k})] \\
&= -\mathrm{E}_{q(x_k)q(\pi)} [x_k \ln(\pi) + (1-x_k) \ln(1-\pi)] \\
&= -\mathrm{E}_{q(x_k)}[x_k] \mathrm{E}_{q(\pi)} [\ln(\pi)] - (1-\mathrm{E}_{q(x_k)}[x_k]) \mathrm{E}_{q(\pi)}[\ln(1-\pi)]
\end{aligned}$$

Which is what we implemented below. Note that `mean(mirrorlog, q(x))` is equal to $\mathrm{E}_{q(x)}[1-\log{x}]$.

```@example create-node
@average_energy Bernoulli (q_out::Any, q_π::Any) = -mean(q_out) * mean(log, q_π) - (1.0 - mean(q_out)) * mean(mirrorlog, q_π)
```



In the case that the interfaces do not factorize, we would get something like `@average_energy MyBernoulli (q_out_π) ...`.


## Using our node in a model


With all the necessary functions defined, we can proceed to test our custom node in an experiment. For this experiment, we will generate a dataset from a Bernoulli distribution with a fixed success probability of 0.75. Next, we will define a probabilistic model that has a `Beta` prior and a `MyBernoulli` likelihood. The `Beta` prior will be used to model our prior belief about the probability of success. The `MyBernoulli` likelihood will be used to model the generative process of the observed data. We start by generating the dataset:

```@example create-node
using Random

rng = MersenneTwister(42)
n = 500
π_real = 0.75
distribution = Bernoulli(π_real)

dataset = float.(rand(rng, distribution, n))

nothing # hide
```



Next, we define our model. Note that we use the `MyBernoulli` node in the model. The model consists of a single latent variable `π`, which has a `Beta` prior and is the parameter of the `MyBernoulli` likelihood. The `MyBernoulli` node takes the value of `π` as its parameter and returns a binary observation. We set the hyperparameters of the `Beta` prior to be 4 and 8, respectively, which correspond to a distribution slightly biased towards higher values of `π`. The model is defined as follows:

```@example create-node
@model function coin_model_mybernoulli(n)

    # `datavar` creates data 'inputs' in our model
    y = datavar(Float64, n)

    # We endow θ parameter of our model with some prior
    π ~ Beta(4.0, 8.0)

    # We assume that outcome of each coin flip is governed by the MyBernoulli distribution
    for i in 1:n
        y[i] ~ MyBernoulli(π)
    end

end
```



Finally, we can run inference with this model and the generated dataset:

```@example create-node
result_mybernoulli = inference(
    model = coin_model_mybernoulli(length(dataset)), 
    data  = (y = dataset, ),
)
```

We have now completed our experiment and obtained the posterior marginal distribution for p through inference. To evaluate the performance of our inference, we can compare the estimated posterior to the true value. In our experiment, the true value for p is 0.75, and we can see that the estimated posterior has a mean of approximately 0.713, which shows that our custom node was able to succesfully pass messages towards the `π` variable in order to learn the true value of the parameter.

```@example create-node
using Plots

rθ = range(0, 1, length = 1000)

p = plot(title = "Inference results")

plot!(rθ, (x) -> pdf(result_mybernoulli.posteriors[:π], x), fillalpha=0.3, fillrange = 0, label="p(π|x)", c=3)
vline!([π_real], label="Real π")
```

As a sanity check, we can create the same model with the `RxInfer` built-in node `Bernoulli` and compare the resulting posterior distribution with the one obtained using our custom `MyBernoulli` node. This will give us confidence that our custom node is working correctly. We use the `Bernoulli` node with the same `Beta` prior and the observed data, and then run inference. We can compare the two posterior distributions and observe that they are exactly the same, which indicates that our custom node is performing as expected.

```@example create-node
@model function coin_model(n)
    
    y = datavar(Float64, n)
    p ~ Beta(4.0, 8.0)

    for i in 1:n
        y[i] ~ Bernoulli(p)
    end

end

result_bernoulli = inference(
    model = coin_model(length(dataset)), 
    data  = (y = dataset, ),
)

if !(result_bernoulli.posteriors[:p] == result_mybernoulli.posteriors[:π])
    error("Results are not identical")
else 
    println("Results are identical 🎉🎉🎉")
end

nothing # hide
```

Congratulations! You have succesfully implemented your own custom node in `RxInfer`. We went through the definition of a node to the implementation of the update rules and marginal posterior calculations. Finally we tested our custom node in a model and checked if we implemented everything correctly.
