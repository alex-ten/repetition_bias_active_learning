using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using LaTeXStrings
using StatsBase


morder = reverse([
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
])

function encode(a::Bool, b::Bool)
    (a ⊽ b) && return 1
    (a & !b) && return 2
    (!a & b) && return 3
    (a & b) && return 4
end


function mysort(df)
    # Group by pid and count case 1 occurrences
    case_1_counts = combine(groupby(df, :pid), :case => (c -> count(==(1), c)) => :case_1_count)
    
    # Sort PIDs by case 1 count in descending order
    sorted_pids = sort(case_1_counts, :case_1_count, rev=true).pid
    
    # Create a dictionary for quick lookup of the new order
    pid_order = Dict(pid => i for (i, pid) in enumerate(sorted_pids))
    
    # Add a new column for sorting
    df.sort_order = [get(pid_order, pid, length(sorted_pids) + 1) for pid in df.pid]
    
    # Sort the dataframe and remove the temporary column
    sorted_df = sort(df, :sort_order)
    select!(sorted_df, Not(:sort_order))
    
    return sorted_df
end


ddf = CSV.read("data/diagnostics_6000.csv", DataFrame)
ddf = unstack(ddf, [:pid, :model, :param], :diag, :value)
ddf.bad_ess = ddf.ess .< 4*100;
ddf.bad_rhat = ddf.rhat .> 1.01;
ddf.bad_both = ddf.bad_ess .& ddf.bad_rhat;
ddf.bad_any = ddf.bad_ess .| ddf.bad_rhat;
@chain ddf begin
    groupby([:pid, :model])
    combine(:bad_ess => any, :bad_rhat => any, :bad_any => any)
    @aside display((_.bad_ess_any .| _.bad_rhat_any) |> sum)
    @aside display(_)
    @aside display(size(_))
    @aside display(_.bad_any_any |> sum)
    groupby(:model)
    combine(:bad_ess_any => sum, :bad_rhat_any => sum, :bad_any_any => sum)
    sort(:bad_any_any_sum)
end



df = @chain ddf begin
    groupby([:pid, :model])
    combine(:bad_ess => any, :bad_rhat => any)
    DataFrames.transform([:bad_ess_any, :bad_rhat_any] => ByRow(encode) => :case)
    mysort()
end


begin
    # Prepare data
    npids = length(unique(df.pid))
    nmodels = length(unique(df.model))
    m = zeros(npids, nmodels)

    for (i, piddf) in enumerate(groupby(df, :pid; sort=false))
        sdf = piddf[sortperm(piddf.model, by=x -> findfirst(==(x), morder)), :]
        for (j, moddf) in enumerate(groupby(sdf, :model; sort=false))
            m[i, j] = moddf.case[1]
        end
    end

    # Axis settings
    lim = (
        x = (.5, npids + .5),
        y = (.5, nmodels + .5)
    )
    axlab = (
        x = "Participant",
        y = "Model"
    )
    α = .7
    mycolors = [(:white, α), (:red, α), (:blue, α), (:purple, α)]

    fig = Figure(size=(900, 300))

    # Legend(fig[1, 1], elements, labels, nbanks=3)
    
    ax = Axis(fig[2, 1], limits=(lim.x, lim.y), xlabel=axlab.x, ylabel=axlab.y)
    ax.xticks = 1:size(m, 1)
    ax.xticklabelsvisible = false
    ax.yticks = (1:size(m, 2), morder)

    hm = heatmap!(ax, m, colormap=mycolors, colorrange=(1, 4))

    # Custom legend in the right panel
    labels = [L"ESS~<~400", L"\hat{R}~>~1.01", L"ESS~<~400~\mathrm{and}~\hat{R}~>~1.01"]
    elements = [PolyElement(polycolor = c) for c in mycolors[2:end]]
    Legend(fig[1, 1], elements, labels, nbanks=1, orientation=:horizontal)
    # save("plots/exclusions.pdf", fig)
    fig
end
