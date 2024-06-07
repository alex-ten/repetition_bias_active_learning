using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using Distributions
using GLMakie
using MLDataUtils: shuffleobs, splitobs, rescale!
using LogExpFunctions
using Turing

#=
z[t-1] = [
    1,
    Y[t-1, 1],
    Y[t-1, 2],
    Y[t-1, 1] * PC[t-1, 1],
    Y[t-1, 2] * PC[t-1, 2],
    Y[t-1, 1] * LP[t-1, 1],
    Y[t-1, 2] * LP[t-1, 2],
    Y[t-1, 1] * Nov[t-1, 1],
    Y[t-1, 2] * Nov[t-1, 2]
]
=#

# Bayesian binary logistic regression
@model function individual_model(z, y)
    n = size(z, 1)
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

    # Priors of intercepts and coefficients
    intercept_1 ~ Normal(0, σ)
    intercept_2 ~ Normal(0, σ)
    intercept_3 ~ Normal(0, σ)
    beta1 ~ MvNormal(Zeros(3), σ^2 * I)
    beta2 ~ MvNormal(Zeros(3), σ^2 * I)
    beta3 ~ MvNormal(Zeros(3), σ^2 * I)

    # Compute the likelihood of the observations
    l1 = intercept1 .+ z * beta1
    l2 = intercept2 .+ z * beta2
    l3 = intercept3 .+ z * beta3
    for i in 1:eachindex(y)
        p = softmax([0, l1[i], l2[i], l3[i]]) # the 0 corresponds to the base category
        y[i] ~ Categorical(p)
    end
end

begin
    pid = unique(df.pid)[5]
    data = @chain df begin
        subset(:pid => ByRow(==(pid)))
        hcat(_.blockTrial, _.famTrial, _.correct)
    end

    plot_data(data)
    trainset, testset = splitobs(shuffleobs(data; obsdim=1), at=1/2, obsdim=1)

    # Turing requires data in matrix and vector form
    train_features = trainset[:, 1:2]
    test_features = testset[:, 1:2]
    train_target = trainset[:, 3]
    test_target = testset[:, 3]

    # Inference
    setprogress!(false)
    m = naive_model(train_features, train_target)
    chain = sample(m, NUTS(), MCMCThreads(), 1_500, 3)

    # Calculate accuracy for our test set
    mean(1 .- abs.(prediction(train_features, chain) .- trainset[:, end])) |> println
    mean(1 .- abs.(prediction(test_features, chain) .- testset[:, end])) |> println
end