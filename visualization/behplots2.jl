using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using ColorSchemes
using CSV
using DataFrames, DataFramesMeta
using Distributions
using GLM
using RollingFunctions
using StatsBase

using Revise
includet("../vislib.jl")

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

function add_suffix(df, suffix)
    rename(df, [name => Symbol(name, "_", suffix) for name in names(df)])
end

function add_regression_line!(ax, model, x, color)
    # Generate points for the regression line
    xrange = range(minimum(x), maximum(x), length=100)
    newdata = DataFrame(Symbol(coefnames(model)[2]) => xrange, Symbol("(Intercept)") => ones(100))
    ypred = predict(model, newdata)
    ypred = [ismissing(y) ? NaN : y for y in ypred]
    ypred = filter(x -> !isnan(x), ypred)

    # Plot regression line and error margins
    lines!(ax, xrange, ypred, color=color, linewidth=1.5)
end

function myc(a=1.0)
    return [Makie.wong_colors(a)[1], Makie.wong_colors(a)[2], (:red, a), Makie.wong_colors(a)[7], Makie.wong_colors(a)[5]]
end

function simbehplot(dfs1, dfs2, models; humcolor=:black)
    set_theme!()
    
    markers = [:circle, :rect, :utriangle, :star4]
    
    fig = Figure(size=(1000, 600))
    
    # Create grid layout with column 3 taking 1/3 of space
    gl = fig[1:2, 1:2] = GridLayout()
    gl_right = fig[1:2, 3] = GridLayout()
    colsize!(fig.layout, 3, Relative(.36))
    
    # Store axes for linking
    axesA = Vector{Axis}(undef, 2)
    axesB = Vector{Axis}(undef, 2)
    
    for (i, v) in ["numSwitches", "blockSizeVarLog"] |> enumerate
        # Row 1: Density plots (Panel A)
        axA = Axis(gl[1, i],
            title = i == 1 ? "Number of switches" : "(log) Block size variance",
            palette = (patchcolor=[(humcolor, .1), (:green, .1)], ),
            ylabel = "Density",
            xticklabelsvisible = false,
            xticksvisible = false
        )
        axesA[i] = axA
        i == 1 && Label(gl[1, i, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
        
        density!(axA, dfs1.hum[:, v], strokecolor = (humcolor, .8), strokewidth=2, boundary= i == 1 ? (0, 180) : (-1.1, 4.1), label="Human")
        # density!(axA, dfs1.rand[:, v], strokecolor = (:green, .8), strokewidth=2, boundary= i == 1 ? (0, 180) : (-1.1, 4.1), label="Random")
        vlines!(axA, i==1 ? 119.3 : log10(0.75), linewidth=2, color=:green)
        for (mi, ml) in enumerate(models)
            density!(axA, @subset(dfs1.sim, :model .== ml)[:, v], color=myc(.1)[mi], strokecolor=myc(.8)[mi], strokewidth=2, boundary= i == 1 ? (0, 180) : (-1.1, 4.1), label=ml)
        end
        
        # Row 2: Scatter plots (Panel B)
        axB = Axis(gl[2, i],
            xlabel="Human data",
            ylabel = "Simulated data",
            title = i == 1 ? "Number of switches" : "(log) Block size variance"
        )
        axesB[i] = axB
        i == 1 && Label(gl[2, i, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
        
        for (j, m) in models |> enumerate
            dfm = innerjoin(
                add_suffix(dfs1.hum, "hum"),
                add_suffix(@subset(dfs1.sim, :model .== m), "sim"),
                on = :pid_hum => :pid_sim
            )
            x = "$v" * "_hum"
            y = "$v" * "_sim"
            scatter!(axB, dfm[:, x], dfm[:, y], color=myc(.2)[j], markersize=6, label=models[j], strokecolor=myc(.8)[j], strokewidth=.8)
            add_regression_line!(
                axB,
                lm(i == 1 ? @formula(numSwitches_sim ~ 1 + numSwitches_hum) : @formula(blockSizeVarLog_sim ~ blockSizeVarLog_hum), dfm),
                dfm[:, x],
                myc(.8)[j]
            )
        end
        ablines!(axB, [0], [1], color=(:black, .4), linestyle=:dash)
    end
    
    # Link x-axes between A and B panels
    linkxaxes!(axesA[1], axesB[1])
    linkxaxes!(axesA[2], axesB[2])
    
    # Panel C: Line plot
    axC = Axis(gl_right[1, 1], xlabel="Trial", ylabel="Proportion switching (smoothed)")
    Label(gl_right[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
    
    lw = 2
    smoothing = 1
    lines!(axC, rollmean(dfs2[:hum].pSwitch, smoothing), linewidth=lw, color=(humcolor, .8), label="Human")
    # lines!(axC, rollmean(dfs2[:rand].pSwitch, smoothing), linewidth=lw, color=(:green, .8), label="Random")
    hlines!(axC, 2/3 ,linewidth=lw, color=:green)
    for (mi, ml) in enumerate(models)
        lines!(axC, rollmean(@subset(dfs2[:sim], :model .== ml).pSwitch, smoothing), linewidth=lw, color=myc(.6)[mi], label=ml)
    end
    
    # Legend
    legend_entries = [
        [PolyElement(color=(humcolor, .8)), "Human data"],
        [LineElement(linewidth=2, color=(:green, .8)), "Random model"],
    ]
    for (mi, ml) in enumerate(models)
        push!(legend_entries, [PolyElement(color=myc(.8)[mi]), ml])
    end
    
    Legend(gl_right[2, 1],
        [entry[1] for entry in legend_entries],
        [entry[2] for entry in legend_entries],
        framevisible = false,
        tellwidth = false
    )

    return fig
end

function sim_rand(p)
    X = rand(Distributions.Categorical(p), 180)
    switch = X[1:length(X)-1] .!= X[2:length(X)]
    blocks = cumsum(switch)
    blockSizeVar = blocks |> countmap |> values |> var
    return sum(switch), blockSizeVar, switch
end

function prep_plot_data()

    hum_data = rename(CSV.read("visualization/plot_df_hum.csv", DataFrame), :trialsComplete => :t)
    pvecs = @chain begin
        @by(hum_data, :pid, 
            :p1 = sum(:famCode .== 1) / 180,
            :p2 = sum(:famCode .== 3) / 180,
            :p3 = sum(:famCode .== 4) / 180,
        )
        @subset(
            :p1 .!= 0,
            :p2 .!= 0,
            :p3 .!= 0,
        )
        @subset(:p1 + :p2 + :p3 .== 1.0)
    end

    df_rand = DataFrame(pid=Int[], numSwitches=Int[], blockSizeVar=Float64[])
    switchmat = BitMatrix(undef, nrow(pvecs), 179)
    for i in 1:nrow(pvecs)
        p = pvecs[i, 2:end] |> collect
        a, b, c = sim_rand(p)
        switchmat[i, :] = c
        push!(df_rand, [i, a, b])
    end

    # df_rand = @chain CSV.read("visualization/plot_sum_df_rand.csv", DataFrame) @transform(:blockSizeVarLog = log10.(:blockSizeVar))
    df_rand = @chain df_rand @transform(:blockSizeVarLog = log10.(:blockSizeVar))
    df_hum = @chain CSV.read("visualization/plot_sum_df_hum.csv", DataFrame) @transform(:blockSizeVarLog = log10.(:blockSizeVar)) @subset(.!isinf.(:blockSizeVarLog))
    df_sim = @chain CSV.read("visualization/plot_sum_df_sim.csv", DataFrame) @transform(:blockSizeVarLog = log10.(:blockSizeVar)) @subset(.!isinf.(:blockSizeVarLog))
    
    dfs1 = (
        rand = df_rand,
        hum = df_hum,
        sim = df_sim
    )

    dfs2 = (
        # rand = CSV.read("visualization/plot_df_rand.csv", DataFrame) |> prep_data,
        rand = DataFrame(t=2:180, pSwitch=mean(switchmat[:, 1:end], dims=1) |> vec),
        sim = @subset(CSV.read("visualization/plot_df_sim.csv", DataFrame)) |> prep_sim_data,
        hum = hum_data |> prep_data
    )

    return dfs1, dfs2
end

dfs1, dfs2 = prep_plot_data();
fig = simbehplot(dfs1, dfs2, 
    [
        "REP+NOV+PC+LP",
        "NOV+PC+LP",
        "REP"
        # "PC+LP",df1.
        # "NOV"
    ]
)
save("plots/sim_beh_sum_choices_switches3_filtered.pdf", fig)