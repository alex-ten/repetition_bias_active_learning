using Pkg; Pkg.activate("Blocking")
using Distributions
using GLMakie
using MLDataUtils: shuffleobs, splitobs, rescale!
using LogExpFunctions
using Turing

function generate_data(θ::Real, T::Int; standardize::Bool=true)
    b₀ = rand(Exponential(θ))
    b₁ = rand(Exponential(θ))
    b₂ = rand(Exponential(θ))

    # Generate blocked choices
    famInd = vcat([fill(rand(1:4), rand(1:10)) for _ in 1:T]...)[1:T]

    # Identify trials within blocks
    blockTrial = ones(Float64, T)
    for t in 2:T
        blockTrial[t] = famInd[t - 1] == famInd[t] ? blockTrial[t - 1] + 1 : 1
    end

    # Identify trials within families
    famTrial = ones(Float64, T)
    for fix in unique(famInd)
        boolInd = famInd .== fix
        famTrial[boolInd] .= collect(1:sum(boolInd))
    end

    m, s = Float64[], Float64[]
    if standardize
        for featureVec in [blockTrial, famTrial]
            m_i, s_i = rescale!(featureVec)
            push!(m, m_i)
            push!(s, s_i)
        end
    end

    # Generate responses based on blockTrials anf famTrials
    y = zeros(Int, T)
    for t in 1:T
        p = logistic(b₀ + b₁ * blockTrial[t] + b₂ * famTrial[t])
        y[t] = rand(Bernoulli(p))
    end
    return hcat(b₀, b₁, b₂), hcat(blockTrial, famTrial, y), (m, s)
end

function plot_data(data::Matrix)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for cat in 0:1
        subdata = data[data[:, 3] .== cat, :]
        jit = (rand(size(subdata)...) .- 0.5) ./ 50
        scatter!(ax, subdata[:, 1] + jit[:, 1], subdata[:, 2] + jit[:, 2], color=ifelse(cat |> Bool, :green, :red), markersize=5)
    end
    fig
end

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
    Ŷ = ifelse.(Y .> 0.5, 1.0, 0.0)
    return Ŷ
end

b, data, scale = generate_data(1, 300)
plot_data(data)
trainset, testset = splitobs(shuffleobs(data; obsdim=1), at=0.7, obsdim=1)

# Turing requires data in matrix and vector form
train_features = trainset[:, 1:2]
test_features = testset[:, 1:2]
train_target = trainset[:, 3]
test_target = testset[:, 3]

# Inference
setprogress!(false)
m = naive_model(train_features, train_target)
chain = sample(m, NUTS(), MCMCThreads(), 1_500, 3)
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set
mean(predictions .== testset[:, end])