using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions

function list_models(sep)
    indices = [1, 2, 3, 4]
    combos = unique!([sort([i for (j,i) in enumerate(indices) if (k & (1 << (j-1))) != 0]) for k in 1:(1<<length(indices))-1])
    vars = ["REP", "NOV", "PC", "LP"]
    return [join(vars[combo], sep) for combo in combos]
end

list_models("+")

function se_margin(se, interval_width=0.95)
    z = quantile(Normal(), 1 - (1 - interval_width) / 2)
    return z * se
end

morder_full = [
    "REP+NOV+PC+LP",
    "REP+NOV+PC",
    "REP+NOV+LP",
    "REP+PC+LP",
    "REP+NOV",
    "REP+PC",
    "REP+LP",
    "REP",
    "NOV+PC+LP",
    "NOV+PC",
    "NOV+LP",
    "PC+LP",
    "NOV",
    "PC",
    "LP"
]

# Δ WAIC
function mcplot()
    method = "waic"
    morder, figsize, lims, save_as = morder_full, (600, 650), (nothing, nothing), "plots/comparisons_full_$(method)_filtered.pdf"
    filepath = "data/comparisons.csv"
    filepath = "data/comparisons_6000_converged_ess100x4_rhat101.csv"

    df = @chain CSV.read(filepath, DataFrame) begin
        @subset(:method .== method)
        @transform(:elpd_margin = se_margin.(:elpd_mcse))
    end

    fig = Figure(size=figsize)
    yticks = collect(1:length(morder))
    ax = Axis(fig[1, 1], 
        yticks = (yticks, morder |> reverse),
        xlabel = "WAIC",
        limits = lims,
        # xscale = log10,
        palette = (patchcolor=[Makie.wong_colors(.3)[1] for i in 15], )
    )
    for (i, m) in zip(yticks, morder |> reverse)
        subdf = @chain df begin
            @subset(:name .== m)
        end
        x = subdf.elpd .+ subdf.elpd_margin
        N = nrow(subdf)
        jitter = collect(LinRange(-.2, .2, N))
        y = fill(i, N)

        boxplot!(ax, y, x;  orientation=:horizontal, show_outliers=false, whiskercolor=(Makie.wong_colors()[1], .8), mediancolor=(:red, 1.0), medianlinewidth=1)
        scatter!(ax, x, y .+ jitter, markersize=5, color=(:black, .3))

    end
    save(save_as, fig)
    fig
end

mcplot()