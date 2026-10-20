using Pkg; Pkg.activate("Blocking")
using GLMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions

function ls_models(sep)
    indices = [1, 2, 3, 4]
    combos = unique!([sort([i for (j,i) in enumerate(indices) if (k & (1 << (j-1))) != 0]) for k in 1:(1<<length(indices))-1])
    vars = ["REP", "NOV", "PC", "LP"]
    return [join(vars[combo], sep) for combo in combos]
end

ls_models("+")

df = @chain CSV.read("data/comparisons.csv", DataFrame) begin
    @subset(:method .== "waic")
end

morder_full = [
    "REP+NOV+PC+LP",
    "NOV+PC+LP",
    "REP",
    "PC+LP",
    "REP+NOV+PC",
    "REP+NOV+LP",
    "REP+PC+LP",
    "REP+NOV",
    "REP+PC",
    "REP+LP",
    "NOV+PC",
    "NOV+LP",
    "NOV",
    "PC",
    "LP"
]

morder_rep = [
    "REP+NOV+PC+LP",
    "REP+NOV+PC",
    "REP+NOV+LP",
    "REP+PC+LP",
    "REP+NOV",
    "REP+PC",
    "REP+LP",
    "REP",
]

morder = morder_full

df_ranks = @chain df begin
    groupby([:rank, :name])
    combine(nrow => :count)
    leftjoin(_, combine(groupby(_, :rank), :count => sum => :total), on = :rank)
    DataFrames.transform([:count, :total] => ByRow((c, t) -> c / t) => :proportion)
end
df_ranks.num = 16 .- [findfirst(x -> x == l, morder) for l in df_ranks.name]
order1 = @chain @subset(df_ranks, :rank .== 1) begin
    sort(:count)
end
df_ranks |> vscodedisplay

function f()
    fig = Figure()
    yticks = collect(1:length(morder))
    ax = Axis(fig[1, 1], 
        title = "Model comparison",
        yticks = (1:length(morder), morder |> reverse),
        xlabel = "Rank proportion",
        limits = ((0, 1), nothing)
    )
    bp = barplot!(ax, Int.(df_ranks.num), df_ranks.proportion,
        stack = df_ranks.rank,
        color = df_ranks.rank,
        direction = :x,
        colormap = Makie.Categorical(:tab20c)
    )
    Colorbar(fig[1, 2], bp, ticks=(1:15, ["$t" for t in 1:15]), label="Rank")

    fig
end

f()