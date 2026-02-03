using Pkg; Pkg.activate("Blocking")
using ColorSchemes
using CSV
using Chain
using DataFrames, DataFramesMeta
using CairoMakie
using RollingFunctions


models = [
    "REP+NOV+PC+LP",
    "NOV+PC+LP"
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
    
    dfs1 = (
        rand = df_rand,
        hum = df_hum,
        sim = df_sim
    )

    dfs2 = (
        rand = CSV.read("visualization/plot_df_rand.csv", DataFrame) |> prep_data,
        sim = @subset(CSV.read("visualization/plot_df_sim.csv", DataFrame)) |> prep_sim_data,
        hum = rename(CSV.read("visualization/plot_df_hum.csv", DataFrame), :trialsComplete => :t) |> prep_data
    )
end


function prep_data(df::DataFrame)
    return @chain df begin
        groupby(:t)
        combine(:switch => mean => :pSwitch)
        subset(:t => ByRow(>(1)))
    end
end

function prep_sim_data(df::DataFrame)
    return @chain df begin
        groupby([:t, :model])
        combine(:switch => mean => :pSwitch)
        subset(:t => ByRow(>(1)))
    end
end

begin
    fig = Figure(size=(800,600))
    alt = 4
    for (i, m) in models |> enumerate
        gtop = fig[1, i] = GridLayout()
        gbot = fig[2, i] = GridLayout()

        # Histograms
        ax11 = Axis(gtop[1, 1], palette = (patchcolor = Makie.wong_colors(.6)[[1, 2, i == 1 ? 3 : alt]], ), xticklabelsvisible=false)
        ax22 = Axis(gtop[2, 2], palette = (patchcolor = Makie.wong_colors(.6)[[1, 2, i == 1 ? 3 : alt]], ), yticklabelsvisible=false)
    
        # Scatterplots
        log10_ = Makie.rich("log", subscript("10"))
        ax21 = Axis(gtop[2, 1], xlabel="Number of switches", ylabel="(log) Block size variance", palette = (color = Makie.wong_colors(.6)[[1, 2, i == 1 ? 3 : alt]], ))
    
        linkyaxes!(ax21, ax22)
        linkxaxes!(ax21, ax11)
    
        # Line plots
        axlp = Axis(gbot[1, 1], title="Proportion switching (smoothed over 10 trials)", xlabel="Trial", palette = (color = Makie.wong_colors(.6)[[1, 2, i == 1 ? 3 : alt]], ))

        for k in dfs |> keys
            df1 = k == :sim ? @subset(dfs1[k], :model .== m) : dfs1[k]
            df2 = k == :sim ? @subset(dfs2[k], :model .== m) : dfs2[k]
            label = k == :hum ? "Human data" : (k == :sim ? "$m" : "Random model")
            density!(ax11, df1.numSwitches, label=label, strokewidth=1, strokecolor=:white, boundary=(0, 180))
            density!(ax22, df1.blockSizeVarLog, direction=:y, strokewidth=1, strokecolor=:white, boundary=(-1.5, 5))
            scatter!(ax21, df1.numSwitches, df1.blockSizeVarLog, alpha=.4, markersize=7)
            lines!(axlp, rollmean(df2.pSwitch, 10), linewidth=2)
        end
        gtop[1, 2] = Legend(fig, ax11, framevisible=false)

        Label(gtop[1, 1, TopLeft()], i == 1 ? "A" : "B",
            fontsize = 26,
            font = :bold,
            padding = (0, 5, 5, 0),
            halign = :right
        )
    end
    rowsize!(fig.layout, 1, Relative(2/3))
    display(fig)
    save("plots/sim_beh_sum_choices_switches.pdf", fig)
end
