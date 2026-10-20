using Pkg; Pkg.activate("Blocking")
using CairoMakie
using CSV
using Chain
using DataFrames, DataFramesMeta
using RollingFunctions
using StatsBase
using Statistics

models = [
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

function prep_data(df::DataFrame)
    return @chain df begin
        groupby(:t)
        combine(:switch => mean => :pSwitch)
        subset(:t => ByRow(>(1)))
    end
end

for m in models
    df_rand = CSV.read("visualization/plot_df_rand.csv", DataFrame) |> prep_data
    df_sim = @subset(CSV.read("visualization/plot_df_sim.csv", DataFrame), :model .== m) |> prep_data
    df_hum = rename(CSV.read("visualization/plot_df_hum.csv", DataFrame), :trialsComplete => :t) |> prep_data

    begin
        fig = Figure()
        ax = Axis(fig[1, 1], title="Proportion switching (smoothed)", xlabel="Trial")
        lines!(ax, rollmean(df_rand.pSwitch, 10), linewidth=3, label="Random model")
        lines!(ax, rollmean(df_hum.pSwitch, 10), linewidth=3, label="Human data")
        lines!(ax, rollmean(df_sim.pSwitch, 10), linewidth=3, label="PC + |LP| model")
        fig
    end

    l = replace(m, "+" => "_")
    # save("plots/sim_beh_sum_$(l).pdf", fig)
    save("plots/sim_switch_$l.pdf", fig)
end