using Pkg; Pkg.activate("Blocking")
# using CairoMakie
using GLMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions

list_models("+")

function se_margin(se, interval_width=0.95)
    z = quantile(Normal(), 1 - (1 - interval_width) / 2)
    return z * se
end

morder_full = [
    "REP+NOV+PC+LP",
    "PC+LP",
    "NOV",
    "REP",
    "REP+NOV+PC",
    "REP+NOV+LP",
    "REP+PC+LP",
    "REP+NOV",
    "REP+PC",
    "REP+LP",
    "NOV+PC+LP",
    "NOV+PC",
    "NOV+LP",
    "PC",
    "LP"
]

# pids_ = unique(df.pid)

# Δ WAIC
begin
    method = "waic"
    morder, figsize, lims, save_as = morder_full, (700, 620), ((10e-2, 10e2), nothing), "plots/comparisons_full_$(method)_by_numswitch.pdf"
    morder, figsize, lims, save_as = morder_full, (700, 500), (nothing, nothing), "plots/comparisons_full_$(method)_by_numswitch.pdf"
    filepath = "data/comparisons.csv"
    filepath = "data/comparisons_6000_converged_ess100x4_rhat101.csv"

    hdf = @chain CSV.read("visualization/plot_sum_df_hum.csv", DataFrame) begin
        select([:pid, :numSwitches, :blockSizeVar, :spEnt])
        @transform(@byrow :blockSizeVarLog = log10(:blockSizeVar))
    end
    
    df = @chain CSV.read(filepath, DataFrame) begin
        @subset(:method .== method)
        @transform(@byrow :elpd_diff_margin = se_margin(:elpd_diff_mcse, .95))
        @transform(@byrow :elpd_margin = se_margin(:elpd_mcse, .95))
        innerjoin(hdf, on=:pid)
        @subset(:numSwitches .> 3, :numSwitches .< 100)
    end

    xvar = :elpd
    cvar = :numSwitches

    fig = Figure(size=figsize)
    yticks = collect(1:length(morder))
    ax = Axis(fig[1, 1], 
        yticks = (yticks, morder |> reverse),
        xlabel = xvar == :elpd_diff ? "Δ WAIC + 95% CI (upper bound)" : "WAIC",
        limits = lims,
        # xscale = log10,
        palette = (patchcolor=[Makie.wong_colors(.3)[1] for i in 15], )
    )
    for (i, m) in zip(yticks, morder |> reverse)
        subdf = @chain df begin
            @subset(:name .== m)
            @aside i == 1 && display(_)
            select([xvar, Symbol("$(xvar)_margin"), cvar, :pid])
        end
        x = subdf[:, xvar] .- subdf[:, Symbol("$(xvar)_margin")]
        N = nrow(subdf)
        jitter = collect(LinRange(-.25, .25, N))
        y = fill(i, N)
        c = subdf[:, cvar]

        scatter!(ax, x, y .+ jitter, markersize=8, color=c, colorrange=extrema(hdf[:, cvar]), alpha=.8, strokecolor=:white, strokewidth=.5)#(:black, .3))
        scatter!(ax, x, y .+ jitter, markersize=8, color=c, colorrange=(3, 100), alpha=.8, strokecolor=:white, strokewidth=.5)#(:black, .3))

    end
    # Colorbar(fig[1, 2], limits=extrema(hdf[:, cvar]), label="Number of switches")
    Colorbar(fig[1, 2], limits=(3, 100), label="Number of switches")
    # save(save_as, fig)
    fig
end
