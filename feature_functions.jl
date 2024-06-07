using Pkg; Pkg.activate("Blocking")
using GLMakie
using Random
using Revise

using Blocking

# Novelty
begin
    Random.seed!(1)
    m = 2
    n = 40

    ix = rand(1:m, n - 1)
    y = zeros(m, n)
    for (j, i) in ix |> enumerate
        y[i, j + 1] = 1.0
    end
    counts = cumsum(y; dims=2)
    mnov = hcat(map(novelty, eachcol(counts))...)
    
    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel="time", ylabel="Count")
    ax2 = Axis(fig[2, 1], xlabel="time", ylabel="Novelty")
    for (c, v) in zip(eachrow(counts), eachrow(mnov))
        println(v)
        lines!(ax1, c)
        lines!(ax2, v)
    end
    fig
end