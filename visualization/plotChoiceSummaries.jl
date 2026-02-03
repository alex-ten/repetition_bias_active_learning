using Pkg; Pkg.activate("Blocking")
using ColorSchemes
using CSV
using Chain
using DataFrames, DataFramesMeta
using CairoMakie

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

begin
    df_rand = CSV.read("visualization/plot_sum_df_rand.csv", DataFrame)
    subset!(df_rand, :blockSizeVar => ByRow(>(0.0)))
    DataFrames.transform!(df_rand, :blockSizeVar => ByRow(log10) => :blockSizeVarLog)
    DataFrames.transform!(df_rand, :spEnt => ByRow(log10) => :spEntLog)

    df_hum = CSV.read("visualization/plot_sum_df_hum.csv", DataFrame)
    subset!(df_hum, :blockSizeVar => ByRow(>(0.0)))
    DataFrames.transform!(df_hum, :blockSizeVar => ByRow(log10) => :blockSizeVarLog)
    DataFrames.transform!(df_hum, :spEnt => ByRow(log10) => :spEntLog)

    # Models
    df_sim = CSV.read("visualization/plot_sum_df_sim.csv", DataFrame)
    subset!(df_sim, :blockSizeVar => ByRow(>(0.0)))
    DataFrames.transform!(df_sim, :blockSizeVar => ByRow(log10) => :blockSizeVarLog)
    DataFrames.transform!(df_sim, :spEnt => ByRow(log10) => :spEntLog)
    
    dfs = (
        rand = df_rand,
        hum = df_hum,
        sim = df_sim
    )
end

for m in models[1:1]
    fig = with_theme(Theme(
        Scatter = (alpha=.3, markersize=10),
        Density = (strokewidth=1, strokecolor=:white),
    )) do 
        fig = Figure()

        # Histograms
        ax11 = Axis(fig[1, 1], palette = (patchcolor = Makie.wong_colors(.6), ))
        ax22 = Axis(fig[2, 2], palette = (patchcolor = Makie.wong_colors(.6), ))

        # Scatterplots
        ax21 = Axis(fig[2, 1], xlabel="Number of switches", ylabel="Block size variance (log10)")

        linkyaxes!(ax21, ax22)
        linkxaxes!(ax21, ax11)

        for k in dfs |> keys
            df = k == :sim ? @subset(dfs[k], :model .== m) : dfs[k]
            label = k == :hum ? "Human data" : (k == :sim ? "$m" : "Random model")
            density!(ax11, df.numSwitches, label=label)
            density!(ax22, df.spEnt, direction=:y)
            scatter!(ax21, df.numSwitches, df.spEnt)
        end
        fig[1, 2] = Legend(fig, ax11, framevisible=false)
        return fig
    end
    l = replace(m, "+" => "_")
    # save("plots/sim_beh_sum_$(l).pdf", fig)
    display(fig)
end

sort(df_hum, :pid)
sort(@subset(df_sim, :model .== "REP+NOV+PC+LP"), :pid)