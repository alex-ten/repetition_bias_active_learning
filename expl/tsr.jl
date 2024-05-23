function onehot(j::Int, m::Int)
    q = m - 1
    oh = zeros(Int, q)
    if j < m
        oh[j] = 1
    end
    return oh
end

function trans(Y::Vector{Bool})
    return 1 - sum(Y)
end

function E(Y, t; F)
    P(Y[t][j] == 1; F)
end

g(t) = cos(2 * Base.π * t / 12)

function mlogit()
end

N = 10
m = 3
q = m - 1
p = 4

# Linear predictor parameters
β = [
    [0.30, 1.25, 0.50, 1.00],
    [-0.20, -2.00, -0.75, -1.00]
]

# Arbitrary initial value
Y₁ = rand(1:m)

# Initialize time series
Y = (
    dum = [BitArray(undef, q) for t in 1:N],
    ref = BitArray(undef, N)
)
p = (
    dum = [Vector{Float16}(undef, q) for t in 1:N],
    ref = Vector{Float16}(undef, N)
)
    
# Encode initial value
if Y₁ == m
    Y.ref[1] = 1 - sum(Y.dum[1])
else
    for j in 1:q
        Y.dum[1][j] = Y₁ == j
    end
end

# Initialize covariates
z = [Vector{Float16}(undef, 4) for t in 1:N]
z[1] = [1.0, g(1), convert(Vector{Float16}, Y.dum[1])'...]

for t in 2:N
    Y.dum[t] = 
end