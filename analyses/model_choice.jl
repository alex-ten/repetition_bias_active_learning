using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using Distributions
using FillArrays
using JLD2
using LinearAlgebra
using LogExpFunctions
using Logging; global_logger(ConsoleLogger(stderr, Logging.Error))
using MLDataUtils: shuffleobs, splitobs, rescale!
using ProgressBars
using StatsPlots
using Turing

m = 3
split_data_at = nothing

function divcols(A, v)
    r = similar(A)
    @inbounds for j = axes(A, 2) 
        @simd for i = axes(A, 1) 
            r[i, j] = v[i] / A[i, j]
        end
    end 
    r
end 

function get_params_matrix(chain, m)
    df = DataFrame(mean(chain))
    # bvec = DataFrame(mean(chain)).mean
    q = m - 1
    d = 1 + q + m * 1
    bmat = Matrix{Float64}(undef, q, d)

    for (j, jsub) in "₁₂" |> enumerate
        # Intercepts
        α = subset(df, :parameters => ByRow(x -> occursin("α$jsub", x |> String)))[:, :mean]

        # Lagged choice params
        β = subset(df, :parameters => ByRow(x -> occursin("β$jsub", x |> String)))[:, :mean]
        
        # Covariate params
        γ = subset(df, :parameters => ByRow(x -> occursin("γ$jsub", x |> String)))[:, :mean]

        bmat[j, :] .= vcat(α, β, γ)
    end
    return bmat
end

function addlag(df, col, lag)
    res = df[:, :]
    res[1:end-lag, col] .= df[1+lag:end, col]
    return res
end

function zscore(col)
    m = mean(col)
    return (col .- m) ./ std(col; corrected=true, mean=m)
end

@model function individual_model_lag(z, y, σ)
    n = size(z, 1)
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

    # Priors of intercepts and coefficients
    β₁₀ ~ Normal(0, σ)
    β₂₀ ~ Normal(0, σ)
    β₁ ~ MvNormal(Zeros(size(z, 2)), σ^2 * I)
    β₂ ~ MvNormal(Zeros(size(z, 2)), σ^2 * I)

    η1 = β₁₀ .+ z * β₁
    η2 = β₂₀ .+ z * β₂
    
    # Compute the likelihood of the observations
    for i in eachindex(y)
        p = softmax([η1[i], η2[i], 0]) # the 0 corresponds to the base category
        y[i] ~ Turing.Categorical(p)
    end
end

# Bayesian multinomial logistic regression
@model function individual_model_lag_covars(z, y, m, σ)
    n = size(z, 1)      # Number of cases (observations)
    q = m - 1           # Number of non-pivot categories
    d = size(z, 2) - q  # Number of non-autoregressive covariates
    length(y) == n ||
        throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

    # Priors of intercepts and coefficients
    α₁ ~ Normal(0, σ)
    α₂ ~ Normal(0, σ)
    β₁ ~ MvNormal(Zeros(q), σ^2 * I)
    β₂ ~ MvNormal(Zeros(q), σ^2 * I)
    γ₁ ~ MvNormal(Zeros(d), σ^2 * I)
    γ₂ ~ MvNormal(Zeros(d), σ^2 * I)

    # Compute the linear predictor for the log odds of each non-pivot category
    η1 = α₁ .+ z[:, 1:q] * β₁ .+ z[:, q+1:end] * γ₁
    η2 = α₂ .+ z[:, 1:q] * β₂ .+ z[:, q+1:end] * γ₂
    
    # Compute the likelihood of the observations
    for i in eachindex(y)
        p = softmax([η1[i], η2[i], 0]) # the 0 corresponds to the base category
        y[i] ~ Turing.Categorical(p)
    end
end

# Bayesian conditional (multinomial) logistic regression
@model function clogit(x, y, σ)
    nr, nc = size(x)    # dimensionality of input matrix
    d = nc ÷ 3          # number of feature groups (e.g., pc1, pc2, pc3 is one group) 
    g = nc ÷ d          # size of each feature group (e.g., pc1, pc2, pc3 is of size 3)
    length(y) == nr || throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

    # Priors of coefficients
    τ ~ Exponential(5)
    wREP ~ Normal(0, σ)
    # wNOV ~ Normal(0, σ)
    # wPC ~ Normal(0, σ)
    # wLP ~ Normal(0, σ)

    # Create params vector
    # β = [wREP, wNOV, wPC, wLP]
    # β = [wPC, wLP]
    # β = [wREP, wNOV]
    # β = [wREP, wPC, wLP]
    β = [wREP]

    # Compute the likelihood of the observations
    z = +([
        β[i] .* x[:, k:k+g-1] 
        for (i, k) in enumerate(1:g:nc)
    ]...)
    for i in eachindex(y)
        p = softmax(z[i, :] ./ τ)
        y[i] ~ Turing.Categorical(p)
    end
end



# Fit
@time begin
    s = "rep"
    df = CSV.read("data/df_features.csv", DataFrame)
    for pid in ProgressBar(unique(df.pid)[16:end], unit="pid")
        dmat = @chain df begin
            # DataFrames.transform(:pid => ByRow(x -> Int(indexin([x], pids)[1])); renamecols=false)
            DataFrames.subset(:pid => ByRow(==(pid))) # 
            # @aside pids_vec = Int.(_.pid)
            addlag(_, :famCode, 1)
            first(nrow(df) - 1)
            # DataFrames.select(:famCode, :c1, :c2, :c3, :nov1, :nov2, :nov3, :pc1, :pc2, :pc3, :lp1, :lp2, :lp3)
            # DataFrames.select(:famCode, :pc1, :pc2, :pc3, :abslp1, :abslp2, :abslp3)
            # DataFrames.select(:famCode, :c1, :c2, :c3, :nov1, :nov2, :nov3)
            # DataFrames.select(:famCode, :c1, :c2, :c3, :pc1, :pc2, :pc3, :abslp1, :abslp2, :abslp3)
            DataFrames.select(:famCode, :c1, :c2, :c3)
            Matrix{Float64}()
        end
        
        trainset, testset = split_data_at |> isnothing ? (dmat, Float64[]) : splitobs(shuffleobs(data; obsdim=1), at=.999, obsdim=1)

        # Turing requires data in matrix and vector form
        train_labels = trainset[:, 1]
        train_features = trainset[:, 2:end]

        if !(testset |> isempty)
            test_labels = testset[:, 1]
            test_features = testset[:, 2:end]
        end

        # Inference
        setprogress!(false)
        model = clogit(train_features, train_labels, 5)
        chain = sample(model, NUTS(500, 0.65), MCMCThreads(), 1000, 3) # (model, sampler, parallel, N, nchains)
        jldsave("data/chains_$(s)/PID-$(pid).jld2"; chain)
    end
end

# StatsPlots.plot(chain)

# Predict choice probabilities for different inputs
# begin
#     # Define inputs (row vectors concatenated into a (transpose of) design matrix)
#     X = [
#         1 0 0;
#         0 1 0;
#         0 0 1;
#     ]

#     # Read a parameter matrix from the chain object (uses MAP estimates)
#     B = get_params_matrix(chain, m)

#     # Compute log odd-ratios of choosing category j in 1:q compared to the pivot (i.e., log(pⱼ / pₘ))
#     η = B * X'

#     # Compute probabilities of choosing category j in 1:q given each input (rows correspond to categories, columns correspond to inputs)
#     p = hcat([
#         exp.(ηⱼ) ./ (1 + sum(exp.(ηⱼ))) 
#         for ηⱼ in eachcol(η)
#     ]...)
    
#     # Compute the probability of choosing category m given each input (rows correspond to categories, columns correspond to inputs)
#     pₘ = [
#         1 / (1 + sum(exp.(ηⱼ)))
#         for ηⱼ in eachcol(η)
#     ]

#     # Concatenate probability vectors (rows correspond to categories, columns correspond to inputs)
#     p = [p; pₘ']
    
#     p |> transpose |> collect |> display
# # meansdf = DataFrame(mean(chain))
# # DataFrames.transform(meansdf, :mean => ByRow(exp) => :exp_mean)
# end

# begin
#     x = X[:, 1:end]
#     m_test = individual_model_lag_hetero_covars(x, Vector{Union{Missing, Float64}}(undef, size(x, 1)), 3, 2, 10)
#     predictions = predict(m_test, chain)
# end

# contab = @chain CSV.read("data/df_features.csv", DataFrame) begin
#     @aside local pid = unique(_.pid)[pidnum]
#     DataFrames.subset(:pid => ByRow(==(pid)))
#     DataFrames.transform(:famCode => (x -> x) => :t)
#     DataFrames.transform(:famCode => (x -> [x[2:end]; NaN]) => Symbol("t+1"))
#     getindex(_, 1:nrow(_) - 1, :)
#     DataFrames.transform(Symbol("t+1") => ByRow(Int); renamecols=false)
#     DataFrames.select(:t, Symbol("t+1"))
#     freqtable(:t, Symbol("t+1"))
#     @aside display(_)
#     prop(; margins=1)
# end

