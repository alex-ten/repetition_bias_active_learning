using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions

function list_models(sep)
    indices = [1, 2, 3, 4]
    combos = unique!([sort([i for (j,i) in enumerate(indices) if (k & (1 << (j-1))) != 0]) for k in 1:(1<<length(indices))-1])
    vars = ["REP", "NOV", "PC", "LP", "RC"]
    return [join(vars[combo], sep) for combo in combos]
end

list_models("+")

function se_margin(se, interval_width=0.95)
    z = quantile(Normal(), 1 - (1 - interval_width) / 2)
    return z * se
end

function boldify(s::AbstractString, pattern::AbstractString)
    pieces = Any[]
    pos = firstindex(s)
    for r in findall(pattern, s)
        first(r) > pos && push!(pieces, String(s[pos:prevind(s, first(r))]))
        push!(pieces, rich(String(s[r]), font = :bold))
        pos = nextind(s, last(r))
    end
    pos <= lastindex(s) && push!(pieces, String(s[pos:end]))
    return rich(pieces...)
end

morder_full = [
    "REP+NOV+PC+LP+RC",
    "REP+NOV+PC+LP",
    "REP+NOV+PC+RC",
    "REP+NOV+LP+RC",
    "REP+NOV+PC",
    "REP+NOV+LP",
    "REP+NOV+RC",
    "REP+PC+LP+RC",
    "REP+PC+LP",
    "REP+PC+RC",
    "REP+LP+RC",
    "REP+NOV",
    "REP+PC",
    "REP+LP",
    "REP+RC",
    "REP",
    "NOV+PC+LP+RC",
    "NOV+PC+LP",
    "NOV+PC+RC",
    "NOV+LP+RC",
    "NOV+PC",
    "NOV+LP",
    "NOV+RC",
    "PC+LP+RC",
    "PC+LP",
    "PC+RC",
    "LP+RC",
    "NOV",
    "PC",
    "LP",
    "RC"
]

# Δ WAIC
function mcplot(morder::Any=missing, method="waic")
    figsize, lims, save_as = (850, 500), (nothing, nothing), "plots/comparisons_full_$(method)_filtered.pdf"
    filepath = "data/comparisons_10k_converged_ess100x4_rhat101.csv"

    df = @chain CSV.read(filepath, DataFrame) begin
        @subset(:method .== method)
        @transform(:elpd_margin = se_margin.(:elpd_mcse))
    end

    if ismissing(morder)
        morder = @chain df begin
            @by(:name, :median_upper_elpd = median(:elpd + :elpd_margin))
            sort(:median_upper_elpd, rev=true)
            getindex(:, :name)
        end
    end

    fig = Figure(size=figsize)
    yticks = collect(1:length(morder))
    ax = Axis(fig[1, 1], 
        xticks = (yticks, boldify.(replace.(morder, "RC" => "CONF"), "REP")),
        xticklabelrotation = pi/2,
        ylabel = uppercase(method),
        limits = lims,
        palette = (patchcolor=[Makie.wong_colors(.3)[1] for i in 15], )
    )

    for (i, m) in zip(yticks, morder)
        subdf = @chain df begin
            @subset(:name .== m)
        end
        x = subdf.elpd .+ subdf.elpd_margin
        N = nrow(subdf)
        jitter = collect(LinRange(-.2, .2, N))
        y = fill(i, N)

        boxplot!(ax, y, x;  orientation=:vertical, show_outliers=false, whiskercolor=(Makie.wong_colors()[1], .8), mediancolor=(:red, 1.0), medianlinewidth=1)
        scatter!(ax, y.+ jitter, x, markersize=5, color=(:black, .3))

    end
    save(save_as, fig)
    fig
end

mcplot(missing, "waic")

@chain CSV.read("data/comparisons_10k_converged_ess100x4_rhat101.csv", DataFrame)


filepath = "data/comparisons_10k_converged_ess100x4_rhat101.csv"
df = @chain CSV.read(filepath, DataFrame) begin
    sort([:pid, :name])
end