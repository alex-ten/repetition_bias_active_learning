using CairoMakie
using CSV
using DataFrames, DataFramesMeta
using Revise

includet("../vislib.jl")

function add_pid_numbers!(df::DataFrame)
    # Get unique PIDs and create a mapping
    unique_pids = unique(df.pid)
    pid_to_num = Dict(pid => i for (i, pid) in enumerate(unique_pids))
    
    # Add new column with mapped values
    df.pidnum = [pid_to_num[pid] for pid in df.pid]
    
    return df
end

function add_rank_column!(df::DataFrame, rank_col::Symbol; ascending::Bool=true)
    # Sort by the ranking column and pid (for tiebreaking)
    # Note: we sort pid in opposite direction of main column to ensure
    # higher pid gets higher rank in case of ties
    sorted_df = sort(df, [rank_col, :pid], rev=[!ascending, ascending])
    
    # Add rank column
    sorted_df.rank = 1:nrow(sorted_df)
    
    # Sort back to original order
    sort!(sorted_df, :pid)
    
    return sorted_df
end

# Human data
function get_dfs()
    df = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
        @subset(:condition .== "free")
        @subset(:stage .== "epochs")
        select([:pid, :trialsComplete, :famInd, :correct])
        rename(:famInd => "famCode")
        groupby(:pid)
        addcols()
        add_pid_numbers!
        # @aside CSV.write("visualization/plot_sum_df_hum.csv", _)
    end
    dfsum = groupby(df, :pid) |> summarize
    return df, dfsum
end


function hdi(x::Vector{<:Real}, prob::Real=0.90)
    sorted_x = sort(x)
    n = length(sorted_x)
    n_included = ceil(Int, prob * n)
    
    # Find the shortest interval containing n_included points
    min_width = Inf
    best_low = sorted_x[1]
    best_high = sorted_x[end]
    
    for i in 1:(n - n_included + 1)
        low = sorted_x[i]
        high = sorted_x[i + n_included - 1]
        width = high - low
        if width < min_width
            min_width = width
            best_low = low
            best_high = high
        end
    end
    
    return (best_low, best_high)
end

function make_plot(df, dfsum, hdi_prob)
    df.famCode .-= 1
    df.famCode[df.famCode .== 0] .= 1
    df.famCode |> unique

    l1 = "Number of Switches"
    l2 = "Block Size Variance"
    l3 = "(log) Block Size Variance"
    fig = Figure(size=(700, 1000))
    
    for (i, ranking) in (:numSwitches, :blockSizeVar) |> enumerate
        dfsum_ = add_rank_column!(dfsum, ranking)
        hl = 0

        # Prepare data
        if ranking == :numSwitches
            hist_data = dfsum[:, ranking]
            dfsum_working = dfsum_
            hl = 179 * (2/3)
        else
            mask = dfsum[:, ranking] .> 0.0
            hist_data = log.(dfsum[mask, ranking])
            dfsum_working = dfsum_[dfsum_[:, ranking] .> 0.0, :]
            hl = log(3/4)
        end
        
        # Compute HDI
        hdi_low, hdi_high = hdi(collect(hist_data), hdi_prob)
        
        # Find ranks of participants inside the HDI
        if ranking == :numSwitches
            in_hdi = dfsum_working[dfsum_working[:, ranking] .>= hdi_low .&& dfsum_working[:, ranking] .<= hdi_high, :]
        else
            log_vals = log.(dfsum_working[:, ranking])
            in_hdi = dfsum_working[log_vals .>= hdi_low .&& log_vals .<= hdi_high, :]
        end
        rank_min = minimum(in_hdi.rank)
        rank_max = maximum(in_hdi.rank)
        
        # Scatter plots in columns 1-2
        ax = Axis(fig[i, 1:2],
            xlabel="Trial number",
            ylabel="Participants",
            leftspinevisible=false, rightspinevisible=false, topspinevisible=false,
            ygridvisible=false, yticksvisible=false, yticklabelsvisible=false,
            limits = ((1, 180), (0, 122)),
            title="Sorted by $(ranking == :numSwitches ? l1 : l2)"
        )
        
        joined_df = outerjoin(df, dfsum_, on=:pid)
        @with joined_df begin
            scatter!(ax, :trialsComplete, :rank, color=:famCode .+ 2, marker=^(:rect), markersize=5, colormap=^(:YlGnBu_3), strokecolor=^(:white))
        end
        
        # Add HDI lines to scatter plot (horizontal lines at rank boundaries)
        hlines!(ax, [rank_min, rank_max], color=:red, linewidth=2)
        hlines!(ax, [rank_min, rank_max], color=:red, linewidth=2)

        # Histograms in column 3 - vertical orientation
        ax2 = Axis(fig[i, 3],
            xlabel="Frequency",
            ylabel="$(ranking == :numSwitches ? l1 : l3)"
        )
        hist!(ax2, hist_data, direction=:x)
        
        # Add HDI lines to histogram (horizontal lines at HDI boundaries)
        hlines!(ax2, [hdi_low, hdi_high], color=:red, linewidth=2)

        # Add expected number of switches and blocksize var
        hlines!(ax2, hl, color=:green, linewidth=2.5)

        x_pos = i == 1 ? 25 : 17.5
        text!(ax2, x_pos, hdi_low, 
            text = string(i == 1 ? Int(hdi_low) : round(hdi_low, digits=2)), 
            color = :red, 
            align = (:right, :bottom),
            offset = (0.0, .60),
            fontsize = 16
        )
        text!(ax2, x_pos, hdi_high, 
            text = string(i == 1 ? Int(hdi_high) : round(hdi_high, digits=2)), 
            color = :red, 
            align = (:right, :bottom),
            offset = (0.0, .60),
            fontsize = 16
        )
    end
    
    # Labels
    Label(fig[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
    Label(fig[2, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
    Label(fig[1, 3, TopLeft()], "C", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
    Label(fig[2, 3, TopLeft()], "D", fontsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
    
    fig
end

fig = make_plot(get_dfs()..., .80)
save("plots/raw_behavior.pdf", fig)
