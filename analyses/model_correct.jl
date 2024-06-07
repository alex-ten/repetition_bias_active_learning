using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using Distributions
using GLMakie
using MLDataUtils: shuffleobs, splitobs, rescale!
using LogExpFunctions
using Turing

# Bayesian binary logistic regression
@model function naive_model(x, y)
    intercept ~ Exponential(1)
    blockTrial ~ Exponential(1)
    famTrial ~ Exponential(1)
    
    for i in eachindex(y)
        v = logistic(intercept + blockTrial * x[i, 1] + famTrial * x[i, 2])
        y[i] ~ Bernoulli(v)
    end
end

function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain, :intercept)
    blockTrial = mean(chain, :blockTrial)
    famTrial = mean(chain, :famTrial)
    
    # Compute the index of the species with the highest probability for each observation.
    Y = logistic.(intercept .+ blockTrial .* x[:, 1] .+ famTrial .* x[:, 2])
    return Y
end

function blocknums(x)
    bn = ones(Float64, length(x))
    sw = x[1:end-1] .!= x[2:end] .|> Int
    bn[2:end] = cumsum(sw) .+ 1
    return bn
end

function cumcount(x)
    return collect(Float64, 1:length(x))
end

function plot_data(data::Matrix)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="blockTrial", ylabel="famTrial")
    for cat in 0:1
        subdata = data[data[:, 3] .== cat, :]
        jit = rand(Uniform(-0.3, 0.3), size(subdata)...)
        scatter!(ax, subdata[:, 1] + jit[:, 1], subdata[:, 2] + jit[:, 2], color=ifelse(cat |> Bool, :green, :red), markersize=5)
    end
    fig
end

# Read data
df = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :correct])
end

# Compute features
df = @chain df begin
    groupby(:pid)
    DataFrames.transform(:famInd => blocknums => :blockInd)
    groupby([:pid, :famInd])
    DataFrames.transform(:famInd => cumcount => :famTrial)
    groupby([:pid, :blockInd])
    DataFrames.transform(:famInd => cumcount => :blockTrial)
    DataFrames.transform(:correct => (x -> Float64.(x)) => :correct)
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