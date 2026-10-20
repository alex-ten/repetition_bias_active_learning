using Pkg; Pkg.activate("Blocking")
using CategoricalArrays: cut
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using Effects
using GLM
using CairoMakie
using JLD2
using MixedModels
using HypothesisTests
using Random
using RollingFunctions
using Statistics
using StatsBase

include("../lib.jl")

function blocknums(x)
    bn = ones(Float64, length(x))
    sw = x[1:end-1] .!= x[2:end] .|> Int
    bn[2:end] = cumsum(sw) .+ 1
    return Integer.(bn)
end

function cumcount(x)
    return Integer.(collect(Float64, 1:length(x)))
end

function zero_and_one(length::Integer)
    z = zeros(Int, length)
    z[end] = 1
    return z
end

function myrunmean(x::AbstractArray, w::Integer)
    try
        return runmean(:confidence, 5)
    catch

    end
end

function switch_confbin(df2::DataFrame)
    # Plot
    df3 = @chain df2 begin
        @transform(
            :confbin = cut(:confidence, 0:0.1:1.0, labels=1:10, extend=true)
        )
        @by([:pid, :confbin], :propLastBlockTrial = mean(:lastBlockTrial))
        @transform(:confbin = parse.(Int, string.(:confbin)) .* 10)
    end

    fig = Figure(size=(800, 500));
    ax = Axis(fig[1, 1], xlabel="Confidence bin (%)", ylabel="Proportion of block-end trials", limits=(nothing, (-0.02, 1.02)), xticks=(5:10:100))
    band!(ax, eff.confidence .* 100, eff.lower, eff.upper, color=(:salmon, 0.15))
    scatter!(ax, df3.confbin .+ rand(length(df3.confbin)) .* 2 .- 6, df3.propLastBlockTrial, color=(:black, .2), markersize=5)
    sp1 = @chain df3 begin
        @by(:confbin, :meanPropLastBlockTrial = mean(:propLastBlockTrial),)
        @with _ begin
            scatter!(ax, :confbin .- 5, :meanPropLastBlockTrial, color=^(:black), marker=^(:diamond))
        end
    end
    sp2 = lines!(ax, eff.confidence .* 100, eff.pred, color=:salmon)
    Legend(
        fig[1, 2],
        [sp1, sp2],
        ["Mean proportion of block-end trials", "Model prediction"]
    )
    return fig
end

function measure_conf_acc_correlations(df, window_sizes::Vector{<:Integer}=Integer[])
    resp = Dict()

    for w in window_sizes
        # Compute features
        df2 = @chain df begin
            @groupby(:pid) # needed in case a person does try do all tasks
            @transform(:famInd2 = encode(:famInd))
            @groupby(:pid)
            @transform(:blockInd = blocknums(:famInd2))
            @groupby([:pid, :famInd2])
            @transform(:famTrial = cumcount(:famInd2))
            @groupby([:pid, :blockInd])
            @transform(:blockTrial = cumcount(:famInd2))
            @transform(:correct = Float64.(:correct))
            @transform(:famInd2 = string.(:famInd2))
            @groupby([:pid, :blockInd])
            @transform(:switch = :blockTrial .== 1 .&& :trialsComplete .!= 1, :lastBlockTrial = zero_and_one(length(:blockTrial)))

            # Running mean of confidence within family
            @groupby([:pid, :famInd])
            @transform(
                :confidence = runmean(:confidence, min(w, length(:confidence))),
                :percentcorrect = runmean(:correct, min(w, length(:correct)))
            )
            @subset(:famTrial .>= w)
        end

        resp[w] = CorrelationTest(df2.confidence, df2.percentcorrect)
    end
    return resp
end

# Read data
df = @chain CSV.read("data/raw_data/combined.csv", DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    # select([:pid, :trialsComplete, :famInd, :correct, :confidence])
end

# Compute features
ws = 5
df2 = @chain df begin
    @groupby(:pid) # needed in case a person does try do all tasks
    @transform(:famInd = encode(:famInd))
    @groupby(:pid)
    @transform(:blockInd = blocknums(:famInd))
    @groupby([:pid, :famInd])
    @transform(:famTrial = cumcount(:famInd))
    @groupby([:pid, :blockInd])
    @transform(:blockTrial = cumcount(:famInd))
    @transform(:correct = Float64.(:correct))
    @transform(:famInd = string.(:famInd))
    @groupby([:pid, :blockInd])
    @transform(:switch = :blockTrial .== 1 .&& :trialsComplete .!= 1, :lastBlockTrial = zero_and_one(length(:blockTrial)))

    # Running mean of confidence within family
    @groupby([:pid, :famInd])
    @transform(
        :confidence = runmean(:confidence, min(ws, length(:confidence)))
    )
    @subset(:famTrial .>= ws)
end

# Confidence vs switching plot
fig = switch_confbin(df, 8)
# save("plots/temp_conf.png", fig)

# Confidence accuracy correlations
wsizes = collect(1:15)
resP = measure_conf_acc_correlations(df, wsizes)


# Model
modresdf = DataFrame(w=Integer[], intercept=Float64[], conf_eff=Float64[], p_intercept=Float64[], p_conf_eff=Float64[])
modresdf2 = DataFrame(w=Integer[], intercept=Float64[], acc_eff=Float64[], p_intercept=Float64[], p_acc_eff=Float64[])
for ws in 1:15
    df2 = @chain df begin
        @groupby(:pid) # needed in case a person does try do all tasks
        @transform(:famInd = encode(:famInd))
        @groupby(:pid)
        @transform(:blockInd = blocknums(:famInd))
        @groupby([:pid, :famInd])
        @transform(:famTrial = cumcount(:famInd))
        @groupby([:pid, :blockInd])
        @transform(:blockTrial = cumcount(:famInd))
        @transform(:correct = Float64.(:correct))
        @transform(:famInd = string.(:famInd))
        @groupby([:pid, :blockInd])
        @transform(:switch = :blockTrial .== 1 .&& :trialsComplete .!= 1, :lastBlockTrial = zero_and_one(length(:blockTrial)))

        # Running mean of confidence within family
        @groupby([:pid, :famInd])
        @transform(
            :confidence = runmean(:confidence, min(ws, length(:confidence))),
            :pc = runmean(:correct, min(ws, length(:correct)))
        )
        @subset(:famTrial .>= ws)
        @groupby(:pid)
        @transform(
            :confidence = zscore(:confidence),
            :pc = zscore(:pc)
        )
    end
    fm = @formula(lastBlockTrial ~ confidence + (1 + confidence | pid))
    mres = glmm(fm, df2, Bernoulli())
    push!(modresdf, [ws, coeftable(mres).cols[1]..., coeftable(mres).cols[4]...])
    fm = @formula(lastBlockTrial ~ pc + (1 + pc | pid))
    mres = glmm(fm, df2, Bernoulli())
    push!(modresdf2, [ws, coeftable(mres).cols[1]..., coeftable(mres).cols[4]...])
end
sort!(modresdf, :w)
sort!(modresdf2, :w)

c = Makie.wong_colors(1)[1:3]
fig = Figure(size=(500, 500));
ax = Axis(fig[2, 1],
    xlabel = "Window size",
    ylabel = "Effect"
)
circs = scatterlines!(ax, wsizes, [resP[k].r for k in wsizes], strokecolor=c[1], strokewidth=1, markercolor=[pvalue(resP[k]) .< 0.01 ? c[1] : :white for k in wsizes], marker=:circle)
diams = scatterlines!(ax, wsizes, modresdf.conf_eff, strokecolor=c[2], strokewidth=1, markercolor=[p .< 0.05 ? c[2] : :white for p in modresdf.p_conf_eff], marker=:diamond)
rects = scatterlines!(ax, wsizes, modresdf2.acc_eff, strokecolor=c[3], strokewidth=1, markercolor=[p .< 0.05 ? c[3] : :white for p in modresdf2.p_acc_eff], marker=:rect)
Legend(fig[1, 1], [circs, rects, diams, MarkerElement(color=:black, marker=' ', markersize=20)], ["Pearson's r between CONF and PC", "Std. effect of PC on switching (log OR)", "Std. effect of CONF on switching (log OR)", "(Filled markers indicate p < .05)"], tellheight=true, tellwidth=false, framevisible=false)
save("plots/competence_defs.pdf", fig)
fig


# Compare specific competence and PC
df2 = @chain df begin
    @groupby(:pid) # needed in case a person does try do all tasks
    @transform(:famInd2 = encode(:famInd))
    @groupby(:pid)
    @transform(:blockInd = blocknums(:famInd2))
    @groupby([:pid, :famInd2])
    @transform(:famTrial = cumcount(:famInd2))
    @groupby([:pid, :blockInd])
    @transform(:blockTrial = cumcount(:famInd2))
    @transform(:correct = Float64.(:correct))
    @transform(:famInd2 = string.(:famInd2))
    @groupby([:pid, :blockInd])
    @transform(:switch = :blockTrial .== 1 .&& :trialsComplete .!= 1, :lastBlockTrial = zero_and_one(length(:blockTrial)))

    # Running mean of confidence within family
    @groupby([:pid, :famInd])
    @transform(
        :confidence = runmean(:confidence, min(4, length(:confidence))),
        :percentcorrect = runmean(:correct, min(15, length(:correct)))
    )
    @subset(:famTrial .>= 15)
    CorrelationTest(_.confidence, _.percentcorrect)
end

