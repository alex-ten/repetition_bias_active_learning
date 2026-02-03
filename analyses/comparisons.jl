using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using GLM
using MixedModels


function se_margin(se, interval_width=0.95)
    z = quantile(Normal(), 1 - (1 - interval_width) / 2)
    return z * se
end

function expandx(x)
    xx = [1, x..., [x[1] * xi for xi in x[2:end]]...]
    return xx
end

function mypredict(mm, x)
    b = coef(mm)
    ci = confint(mm)
    y = sum(b .* expandx(x))
    lo = sum(collect(ci.lower) .* expandx(x))
    hi = sum(collect(ci.upper) .* expandx(x))
    return [y, lo, hi]
end


df = @chain CSV.read("data/comparisons_6000_converged_ess100x4_rhat101.csv", DataFrame) begin
    @subset(:method .== "waic")
    @rtransform(:elpd_margin = se_margin(:elpd_mcse, .95))
    @rtransform(:elpd_diff_margin = se_margin(:elpd_diff_mcse, .95))
    innerjoin(select(CSV.read("visualization/plot_sum_df_hum.csv", DataFrame), [:pid, :numSwitches]), on=:pid)
    @rtransform(:REP = occursin("REP", :name) |> Int)
    @rtransform(:NOV = occursin("NOV", :name) |> Int)
    @rtransform(:PC = occursin("PC", :name) |> Int)
    @rtransform(:LP = occursin("LP", :name) |> Int)
    @select(:pid, :waic = :elpd, :numSwitches, :REP, :NOV, :PC, :LP)
    @aside centered = @transform(combine(groupby(_, :pid), :numSwitches => first => :ns), :numSwitchesCentered = :ns .- mean(:ns))    
    innerjoin(_, centered, on=:pid)
end

dfcent = @chain df begin
    @groupby(:pid)
    @transform(
        :REP = :REP .- mean(:REP),
        :NOV = :NOV .- mean(:NOV),
        :PC = :PC .- mean(:PC),
        :LP = :LP .- mean(:LP)
    )
end

# Analysis 1: Independent effects of predictor inclusions on predictive accuracy
mm = fit(MixedModel, 
    @formula(waic ~ 1 + REP + NOV + PC + LP + (1 + REP + NOV + PC + LP | pid)), 
    dfcent
)
confint(mm; level=.95)

# Analysis 2: 
dfrep = @subset(df, :REP .== 1)
dfrepcent = @chain dfrep begin
    @groupby(:pid)
    @transform(
        :NOV = :NOV .- mean(:NOV),
        :PC = :PC .- mean(:PC),
        :LP = :LP .- mean(:LP)
    )
end
mm2 = fit(MixedModel, 
    @formula(waic ~ 1 + NOV + PC + LP + (1 + NOV + PC + LP | pid)),
    dfrepcent
)
confint(mm2; level=.95)