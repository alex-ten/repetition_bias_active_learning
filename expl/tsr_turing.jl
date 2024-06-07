using Pkg; Pkg.activate("Blocking")
using DataFrames
using FIllArrays: Zeros
using LinearAlgebra: I
using MLDataUtils: shuffleobs, splitobs, rescale!
using NNlib: softmax
using StatsBase
using Turing

g(t) = cos(2 * Base.π * t / 12)

function dummy(j::Int, m::Int)
    q = m - 1
    dum = zeros(Int, q)
    if j < m
        dum[j] = 1
    end
    return dum, Int(j == m)
end

function agresti(ηₜ::Vector{T}) where T <: Real
    return exp.(ηₜ) ./ (1 + sum(exp.(ηₜ)))
end

function prediction(x::Matrix, chain)
    # Pull the means from each parameter's sampled values in the chain.
    intercept1 = mean(chain, :intercept1)
    intercept2 = mean(chain, :intercept2)
    beta1 = [
        mean(chain, k) for k in MCMCChains.namesingroup(chain, :beta1)
    ]
    beta2 = [
        mean(chain, k) for k in MCMCChains.namesingroup(chain, :beta2)
    ]

    # Compute the index of the species with the highest probability for each observation.
    Y1 = intercept1 .+ x * beta1
    Y2 = intercept2 .+ x * beta2
    Ŷ = [
        argmax((0, x, y)) for (x, y) in zip(Y1, Y2)
    ]

    return Ŷ
end

N = 51
m = 3
q = m - 1
p = 4

# Define linear predictor parameters
β = [
    [0.30, 1.25, 0.50, 1.00],
    [-0.20, -2.00, -0.75, -1.00]
]

# Create containers for storing values of Y and p
Y = (
    dum = [BitArray(undef, q) for t in 1:N],
    ref = BitArray(undef, N),
    cix = Vector{Int}(undef, N)
)
p = (
    dum = [Vector{Real}(undef, q) for t in 1:N],
    ref = Vector{Real}(undef, N)
)
 
# Sample the first observation randomly
p.dum[1] = fill(1/m, q)
p.ref[1] = 1/m
cix = sample(1:m, [p.dum[1]..., p.ref[1]] |> Weights)
Y.cix[1] = cix
Y.dum[1], Y.ref[1] = dummy(cix, m)

# Initialize values of the first vector of covariates (1, g(t), Y₁₁, Y₁₂)
z = [Vector{Real}(undef, 4) for t in 1:N]
z[1] = [1.0, g(1), Y.dum[1]...]

# Initialize values for linear predictor
η = [Vector{Real}(undef, q) for t in 1:N]
η[1] .= NaN

# Sample new values from the model
for t in 2:N
    η[t] = [βⱼ' * z[t - 1] for βⱼ in β]
    p.dum[t] = agresti(η[t])
    p.ref[t] = 1 - sum(p.dum[t])
    cix = sample(1:m, [p.dum[t]..., p.ref[t]] |> Weights)
    Y.cix[t] = cix
    Y.dum[t], Y.ref[t] = dummy(cix, m)
    z[t] = [1.0, g(t), Y.dum[t]...]
end

# fig = Figure()
# ax = (
#     a = Axis(fig[1, 1], title="a) Y[t]"),
#     b = Axis(fig[1, 2], title="b) p[t][1]"),
#     c = Axis(fig[2, 1], title="c) p[t][2]"),
#     d = Axis(fig[2, 2], title="d) p[t][3]")
# )

# lines!(ax.a, 1:N, Y.cix)
# lines!(ax.b, 1:N, [v[1] for v in p.dum])
# lines!(ax.c, 1:N, [v[2] for v in p.dum])
# lines!(ax.d, 1:N, p.ref)

# Construct a dataframe
z_matrix = hcat(z...)[:, 2:end]
df = DataFrame(
    x = z_matrix[2, :],
    prev_Y1 = z_matrix[3, :],
    prev_Y2 = z_matrix[4, :],
    Y = Y.cix[2:end]
)
trainset, testset = splitobs(shuffleobs(df), 0.5)
features = [:x, :prev_Y1, :prev_Y2]
numerics = [:x]
target = :Y

# Standardize the features
for feature in numerics
    μ, σ = rescale!(trainset[!, feature]; obsdim=1)
    println(feature)
    println(μ)
    println(σ)
    rescale!(testset[!, feature], μ, σ; obsdim=1)
end

# Turing requires data in matrix and vector form
train_features = Matrix(trainset[!, features])
test_features = Matrix(testset[!, features])
train_target = trainset[!, target]
test_target = testset[!, target]

# Bayesian multinomial logistic regression
@model function logistic_regression(z, y, σ)
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

setprogress!(false)

m = logistic_regression(train_features, train_target, 1)
chain = sample(m, NUTS(), MCMCThreads(), 1_500, 3)
predictions = prediction(test_features, chain)

# Calculate accuracy for our test set
mean(predictions .== testset[!, :Y])

# Accuracy per class
for s in 1:3
    rows = testset[!, :Y] .== s
    println("Number of category $s: $(count(rows))")
    println("Accuracy category $s: $(mean(predictions[rows] .== testset[rows, :Y]))")
end