using Pkg; Pkg.activate("Blocking")
using Distributions
using CairoMakie
using LogExpFunctions

function lcurve(t::Real, γ::Real)
    return 1 - exp(-γ * t) / 2
end

function pc(h::Vector{<:AbstractVector})
    return [mean(last(v, 15)) for v in h]
end

function lp(h::Vector{<:AbstractVector})
    return [abs(first(last(v, 9)) - mean(last(v, 10))) for v in h]
end

function nov(h::Vector{<:AbstractVector})
    k = length(h)
    count_i = [length(v) + 1 for v in h]
    count_sum = sum(count_i) + k
    return -log.(count_i ./ count_sum)
end

function mod_preds(n::Integer, f::Function; τ::Real=1.0, γ::Real=0.05)
    k = 3
    h = [zeros(15) for i in 1:k]
    for i in 1:k
        h[i][1:2:end] .= 1
    end
    ph = []
    ch = []
    for t in 1:n
        pcat = softmax(f(h) ./ τ)
        push!(ph, pcat)
        i = Distributions.Categorical(pcat) |> rand
        push!(ch, i)
        pcorr = lcurve(t, γ)
        y = Bernoulli(pcorr) |> rand
        push!(h[i], y)
    end

    ph = hcat(ph...)
    fig = Figure(size=(900, 400))
    ax = Axis(fig[1, 1], limits=((1, n), (-0.02, 1.02)))
    cmap = to_colormap(:Set1)
    series!(ax, ph, color=:Set1)
    scatter!(ax, 1:n, zero(ch), color=[cmap[i] for i in ch])
    
    return fig
end

mod_preds(180, lp, γ=0.01, τ=.01)