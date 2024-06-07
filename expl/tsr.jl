using Pkg; Pkg.activate("Blocking")
using StatsBase
using GLMakie

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

N = 200
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

fig = Figure()
ax = (
    a = Axis(fig[1, 1], title="a) Y[t]"),
    b = Axis(fig[1, 2], title="b) p[t][1]"),
    c = Axis(fig[2, 1], title="c) p[t][2]"),
    d = Axis(fig[2, 2], title="d) p[t][3]")
)

lines!(ax.a, 1:N, Y.cix)
lines!(ax.b, 1:N, [v[1] for v in p.dum])
lines!(ax.c, 1:N, [v[2] for v in p.dum])
lines!(ax.d, 1:N, p.ref)

function pll(β::Vector{Vector{T}}, Y::NamedTuple, z::Any) where T <: Real
    N = length(z)
    b = hcat(β...)
    obs = vcat(hcat(z...))
    η = b' * obs
    p = hcat([agresti(η[:, t]) for t in 1:N]...)
    p = vcat(p, 1 .- sum(p; dims=1))
    likelihood = [c[ix] for (ix, c) in zip(Y.cix[2:end], eachcol(p[:, 2:end]))]
    return sum(log.(likelihood))
end
