# """Decision-making model"""
# mutable struct DMM()
#     cf::Function
#     function DMM()
#         new()
#     end
# end

"""Novelty function from Xu, Modirshanechi et al. (2020)"""
function novelty(c)::Vector{Float64}
    numer = c .+ 1
    denom = sum(c) + length(c)
    p = numer ./ denom
    return -log.(p)
end

"""Softmax function"""
function mysoftmax(v::Vector{Real}, τ::Real=1.0)::Vector{Real}
    z = v .- maximum(v)
    z = z ./ τ
    eᶻ = exp.(z)
    return eᶻ ./ sum(eᶻ)
end

""""""
@model function choice_logr(z, y, σ)
    n = size(z, 1)
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `x` and `y` is not equal"))

    # Priors of intercepts and coefficients.
    intercept1 ~ Normal(0, σ)
    intercept2 ~ Normal(0, σ)
    beta1 ~ MvNormal(Zeros(3), σ^2 * I)
    beta2 ~ MvNormal(Zeros(3), σ^2 * I)

    # Compute the likelihood of the observations.
    l1 = intercept1 .+ z * beta1
    l2 = intercept2 .+ z * beta2
    for i in 1:n
        # the 0 corresponds to the base category
        v = softmax([0, l1[i], l2[i]])
        y[i] ~ Categorical(v)
    end
end

@model function choice_logr(z, y, σ)
    n = size(z, 1)
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `x` and `y` is not equal"))

    # Priors of intercepts and coefficients.
    intercept1 ~ Normal(0, σ)
    intercept2 ~ Normal(0, σ)
    beta1 ~ MvNormal(Zeros(3), σ^2 * I)
    beta2 ~ MvNormal(Zeros(3), σ^2 * I)

    # Compute the likelihood of the observations.
    l1 = intercept1 .+ z * beta1
    l2 = intercept2 .+ z * beta2
    for i in 1:n
        # the 0 corresponds to the base category
        v = softmax([0, l1[i], l2[i]])
        y[i] ~ Categorical(v)
    end
end

# Bayesian conditional (multinomial) logistic regression
@model function hclogit(x, y, n, pids, σ)
    N, dxm = size(x)
    d = dxm ÷ 3
    length(y) == N || throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

    # Hyper priors
    wREP_hyper ~ Normal(0, σ)
    wNOV_hyper ~ Normal(0, σ)
    wPC_hyper ~ Normal(0, σ)
    wLP_hyper ~ Normal(0, σ)

    # Individual priors
    wREP ~ filldist(Normal(wREP_hyper, σ), n)
    wNOV ~ filldist(Normal(wNOV_hyper, σ), n)
    wPC ~ filldist(Normal(wPC_hyper, σ), n)
    wLP ~ filldist(Normal(wLP_hyper, σ), n)

    # Compute the likelihood of the observations
    for i in eachindex(y)
        j = pids[i]
        z = wREP[j] .* x[i, 1:3] + wNOV[j] .* x[i, 4:6] + wPC[j] .* x[i, 7:9] + wLP[j] .* x[i, 10:12]
        p = softmax(z)
        y[i] ~ Turing.Categorical(p)
    end

    y = arraydist(Categorical(p))

end