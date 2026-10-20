using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFramesMeta
using InvertedIndices
using MixedModels
using Statistics
using SummaryTables
import WriteDocx as W

include("../vislib.jl")

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

function mean_alts(vals, exclude_idx)
    if ismissing(exclude_idx) || exclude_idx < 1 || exclude_idx > length(vals)
        return mean(vals)
    end
    clean_vals = deleteat!(collect(vals), exclude_idx)
    return mean(clean_vals)
end

dff = @chain CSV.read("data/df_features.csv", DataFrame) begin
    @groupby(:pid); @transform(:famInd = encode(:famInd))
    @groupby(:pid); @transform(:blockInd = blocknums(:famInd))
    @groupby([:pid, :famInd]); @transform(:famTrial = cumcount(:famInd))
    @groupby([:pid, :blockInd]); @transform(:blockTrial = cumcount(:famInd))
    @transform(:famInd = string.(:famInd))
    @groupby([:pid, :blockInd]); @transform(:lastBlockTrial = zero_and_one(length(:blockTrial)))
    @rtransform(:lastBlockTrial = ifelse(:trialsComplete == 180, 0, :lastBlockTrial))
    @rsubset(:trialsComplete !== 180)
    @rtransform(:rc = [:rc1, :rc2, :rc3][:famCode])
end

model_data = @chain dff begin
    @rtransform(
        :meanAltNov = mean_alts([:nov1, :nov2, :nov3], :famCode),
        :meanAltPc = mean_alts([:pc1, :pc2, :pc3], :famCode),
        :meanAltLp = mean_alts([:lp1, :lp2, :lp3], :famCode),
        :meanAltRc = mean_alts([:rc1, :rc2, :rc3], :famCode)
    )
    @transform(:logBlockTrial = log.(:blockTrial))
    @select(:pid, :lastBlockTrial, :blockTrial, :logBlockTrial, :nov, :meanAltNov, :pc, :meanAltPc, :lp, :meanAltLp, :rc, :meanAltRc)
    @transform(:novContr = :nov - :meanAltNov)
    @groupby(:pid)
    @transform(
        :nov = :nov .- mean(:nov),
        :pc = :pc .- mean(:pc),
        :lp = :lp .- mean(:lp),
        :rc = :rc .- mean(:rc),
        :meanAltNov = :meanAltNov .- mean(:meanAltNov),
        :meanAltPc = :meanAltPc .- mean(:meanAltPc),
        :meanAltLp = :meanAltLp .- mean(:meanAltLp),
        :meanAltRc = :meanAltRc .- mean(:meanAltRc)
    )
end

display(@select(dff, Not(r"^(c|nov|lp|rc|pc|cp|abslp|abscp)\d", :pc, :lp, :abslp, :nov, :rcCentered, :pcCentered, :lpCentered, :novCentered, :abscp, :abslpCentered)))

mswitch_centered = fit(MixedModel, @formula(lastBlockTrial ~ 
    1  + logBlockTrial + novContr + pc + lp + rc + meanAltPc + meanAltLp + meanAltRc +
    (1 + logBlockTrial + novContr + pc + lp + rc + meanAltPc + meanAltLp + meanAltRc | pid)), model_data, Bernoulli()
)

# Generate table
data = DataFrame(
    predictor = coeftable(mswitch_centered).rownms,
    estimate = round.(coeftable(mswitch_centered).cols[1], digits=2),
    ci95 = cirep(confint(mswitch_centered, level=.95); digits=2),
    p = pvalrep.(coeftable(mswitch_centered).cols[4], digits=3)
)


# Word table
table3 = simple_table(data, [:predictor => "Predictor", :estimate => "Estimate", :ci95 => "95% CI", :p => "p"])

doc = W.Document(
    W.Body([
        W.Section([
            SummaryTables.to_docx(table3)
        ])
    ])
)
W.save(joinpath("plots", "table3.docx"), doc)

# Latex table
function modelrows(m)
    ct = coeftable(m)
    ci = confint(m; level=.95)
    est = ct.cols[1]
    pval = ct.cols[4]
    coef = ci.coef |> collect
    lo = ci.lower |> collect
    hi = ci.upper |> collect
    [
        [
            replace(string(coef[i]), "(Intercept)" => "Intercept"),
            string(round(est[i]; digits=2)),
            cirep(lo[i], hi[i]),
            pvalrep(pval[i], digits=3),
        ]
        for i in eachindex(est)
    ]
end

tbl3 = Any[
    Rule(:h),
    ["Predictor", "Estimate", "95% CI", LaTeX(raw"\textit{p}")],
    Rule(:h),
    modelrows(mswitch_centered)...,
    Rule(:h),
]

latex_tabular(joinpath("plots", "table3.tex"), Tabular("lccc"), tbl3)