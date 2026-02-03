"""Helper function to get 1 subject's df"""
function getsubj(df, i)
    pids = unique(sort(df.pid))
    pid = pids[i]
    return df[df.pid .== pid, :]
end

"""Forward fill function"""
ffill(v) = v[accumulate(max, [i * !isnan(v[i]) for i in eachindex(v)], init=1)]
function mypad(v, n, ones_at)
    p = zeros(n)
    p[ones_at] .= 1
    return vcat(p, v)
end

"""
Encodes a categorical column of arbitrary encoding into a consistent numerical encoding. 
This ensures that categories are properly encoded. I.e., if there are K categories, 
exactly K integers from 1 to K will be assigned to each category respectively.
"""
function encode(x)
    y = x[:]
    u = sort(unique(y))
    for i in eachindex(u)
        y[x .== u[i]] .= i
    end
    return y
end

"""
Expands a categorical encoding into a dummy variable: i.e., a one-hot vector of length K,
where all elements are equal to 0.0, except the x-th element which is equal to 1.0
"""
function dummify(x, K)
    oh = zeros(Float64, K)
    oh[x] = 1.0
    return oh
end

"""Novelty function adapted from Xu, Modirshanechi et al. (2020)."""
function novelty(c, K)
    t = collect(1:length(c))
    numer = cumsum(c) .+ 1
    denom = t .+ K
    p = numer ./ denom
    return -log.(p)
end

function novelty_last(c, K)
    t = length(c)
    numer = sum(c) .+ 1
    denom = t + K
    p = numer / denom
    return -log(p)
end

"""Proportion correct function adapted from Ten et al., (2021)."""
function pc(x; padding=15)
    px = mypad(x, padding, 1:2:padding)
    return [
        mean(px[1+i-padding : i])
        for i in padding + 1:padding + length(x)
    ]
end

"""Learning progress function adapted from Ten et al., (2021)"""
function lp(x, a, b; padding=15)
    px = mypad(x, padding, 1:2:padding)
    return [
        mean(px[1+i-padding : i][1:a]) - mean(px[1+i-padding : i][1-b+padding : end]) 
        for i in padding + 1:padding + length(x)
    ]
end

"""Apply function `f` with `args`` and `kwargs` *partially*, i.e., to slices of `x` that correspond to categories given by cats. 
Then assign the result onto a corresponding slice of a NaN-filled matrix with `ncol` == `m` """
function papply(x, f, cats, m, args...; kwargs...)
    res = fill(NaN, length(x), m)
    for i in 1:m
        res[cats .== i, i] = f(x[cats .== i], args...; kwargs...)
    end
    return res
end


function init_h(id::String, ix::Int)
    return DataFrame(
        pix = Int[ix],
        pid = String[id],
        model = String["NA"],
        t = Int[0],
        famCode = Int[0],
        c1 = Int[0],
        c2 = Int[0],
        c3 = Int[0],
        correct = Int[1]
    )
end

function get_rep(h)
    return h[end, Cols(:c1, :c2, :c3)] |> collect
end

function get_nov(h)
    return mapcols(c -> novelty_last(c, 3), h[:, 4:end])[1, Cols(:c1, :c2, :c3)] |> collect
end

function Δpc(x)
    return mean(x[9:end]) - mean(x[1:10])
end

function get_pc(h)
    return [mean(mypad(h[BitArray(h[:, c]), :correct], 15, 1:2:15)[end-15 + 1:end]) for c in [:c1, :c2, :c3]]
end

function get_lp(h)
    return [Δpc(mypad(h[BitArray(h[:, c]), :correct], 15, 1:2:15)[end-15 + 1:end]) for c in [:c1, :c2, :c3]]
end

function extract_features(h)
    rep = get_rep(h)
    nov = get_nov(h)
    pc_ = get_pc(h)
    lp_ = get_lp(h)
    return cat(rep, nov, pc_, lp_; dims=2)
end

function get_choice(θ, x, Ω)
    u = x * θ[1:end-1]
    p = softmax(u ./ last(θ))
    return StatsBase.sample(Ω, Weights(p))
end

function sample_params(datadir::Any=missing)
    ismissing(datadir) && return zeros(4)
    path = rand(readdir(datadir))
    chain = load(datadir * "/" * path)["chain"]
    d = Dict(
        :τ => 1.0,
        :wREP => 0.0,
        :wNOV => 0.0,
        :wPC => 0.0,
        :wLP => 0.0
    )
    for k in d |> keys
        if k in chain.name_map.parameters
            d[k] = mean(chain)[k, :mean]
        end
    end
    return [d[:wREP], d[:wNOV], d[:wPC], d[:wLP], d[:τ]]
end

function get_fb(y, i, m)
    p = predict(m, DataFrame(correct=[0.0], trialsComplete=[i], famInd=["$y"]))[1]
    return Int(rand() < p)
end

function params_vec(param_keys, param_vals)
    d = Dict(
        "τ" => 1.0,
        "REP" => 0.0,
        "NOV" => 0.0,
        "PC" => 0.0,
        "LP" => 0.0
    )
    for (k, v) in zip(param_keys, param_vals)
        d[k] = v
    end
    return [d["REP"], d["NOV"], d["PC"], d["LP"], d["τ"]]
end