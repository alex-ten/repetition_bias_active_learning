using Turing
using FIllArrays
using NNlib: softmax

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