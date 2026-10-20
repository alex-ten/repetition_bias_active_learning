using Pkg; Pkg.activate("Blocking")
using CairoMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using GLM
using MixedModels
using SummaryTables
import WriteDocx as W
using LaTeXTabulars, LaTeXEscapes, Printf

include("../vislib.jl")

df = @chain CSV.read("data/comparisons_10k_converged_ess100x4_rhat101.csv", DataFrame) begin
    @subset(:method .== "waic")
    @rtransform(:elpd_margin = se_margin(:elpd_mcse, .95))
    @rtransform(:elpd_diff_margin = se_margin(:elpd_diff_mcse, .95))
    innerjoin(select(CSV.read("visualization/plot_sum_df_hum.csv", DataFrame), [:pid, :numSwitches]), on=:pid)
    @rtransform(:REP = occursin("REP", :name) |> Int)
    @rtransform(:NOV = occursin("NOV", :name) |> Int)
    @rtransform(:PC = occursin("PC", :name) |> Int)
    @rtransform(:LP = occursin("LP", :name) |> Int)
    @rtransform(:RC = occursin("RC", :name) |> Int)
    @select(:pid, :waic = :elpd, :numSwitches, :REP, :NOV, :PC, :LP, :RC)
    @aside centered = @transform(combine(groupby(_, :pid), :numSwitches => first => :ns), :numSwitchesCentered = :ns .- mean(:ns))    
    innerjoin(_, centered, on=:pid)
end

dfcent = @chain df begin
    @groupby(:pid)
    @transform(
        :REP = :REP .- mean(:REP),
        :NOV = :NOV .- mean(:NOV),
        :PC = :PC .- mean(:PC),
        :LP = :LP .- mean(:LP),
        :RC = :RC .- mean(:RC)
    )
end

# Analysis 1: Independent effects of predictor inclusions on predictive accuracy
mm = fit(MixedModel, 
    @formula(waic ~ 1 + REP + NOV + PC + LP + RC + (1 + REP + NOV + PC + LP + RC | pid)), 
    dfcent
)
cis = confint(mm; level=.95)

# Analysis 2: 
dfrep = @subset(df, :REP .== 1)
dfrepcent = @chain dfrep begin
    @groupby(:pid)
    @transform(
        :NOV = :NOV .- mean(:NOV),
        :PC = :PC .- mean(:PC),
        :LP = :LP .- mean(:LP),
        :RC = :RC .- mean(:RC)
    )
end
mm2 = fit(MixedModel, 
    @formula(waic ~ 1 + NOV + PC + LP + RC + (1 + NOV + PC + LP + RC| pid)),
    dfrepcent
)
cis2 = confint(mm2; level=.95)

# Generate word table (Table 2)
data = DataFrame(
    models = vcat(fill("All", 6), fill("With REP", 5)),
    predictor = ["Intercept", "REP", "NOV", "PC", "LP", "RC", "Intercept", "NOV", "PC", "LP", "RC"],
    estimate = vcat([round.(coeftable(m).cols[1], digits=2) for m in [mm, mm2]]...),
    ci95 = vcat([cirep(confint(m, level=.95); digits=2) for m in [mm, mm2]]...),
    p = vcat([pvalrep.(coeftable(m).cols[4], digits=3) for m in [mm, mm2]]...)
)

table2 = simple_table(data, [:models => "Scope", :predictor => "Predictor", :estimate => "Estimate", :ci95 => "95% CI", :p => "p"])

doc = W.Document(
    W.Body([
        W.Section([
            SummaryTables.to_docx(table2)
        ])
    ])
)
W.save(joinpath("plots", "table2.docx"), doc)

# Generate latex table
cirep(lo, hi; digits=2) = LaTeX("{[}$(round(lo; digits=digits)), $(round(hi; digits=digits)){]}")

function pvalrep(p; digits=3)
    p < 0.001 && return LaTeX(raw"$<$ .001")
    LaTeX(replace(string(round(p; digits=digits)), r"^0\." => "."))
end

function rows(m, cis)
    est = coeftable(m).cols[1]
    pval = coeftable(m).cols[4]
    coef = cis.coef |> collect
    lo = cis.lower |> collect
    hi = cis.upper |> collect
    [
        [
            "",
            replace(coef[i], "(Intercept)" => "Intercept"),
            string(round(est[i]; digits=2)),
            cirep(lo[i], hi[i]),
            pvalrep(pval[i], digits=3),
        ]
        for i in eachindex(est)
    ]
end

section(t) = [
    MultiColumn(2, :l, LaTeX("\\textbf{$t}")),
    MultiColumn(1, :l, ""),
    MultiColumn(1, :l, ""),
    MultiColumn(1, :l, ""),
]

tbl = Any[
    Rule(:h),
    ["", "", "Estimate", "95% CI", LaTeX(raw"\textit{p}")],
    Rule(:h),
    section("All models"),
    rows(mm, cis)...,
    section("Models with REP"),
    rows(mm2, cis2)...,
    Rule(:h),
]

latex_tabular(joinpath("plots", "table2.tex"), Tabular("llccc"), tbl)