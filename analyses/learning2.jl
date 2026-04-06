using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using MixedModels, GLM, StatsModels
using HypothesisTests
using StatsBase

function blocknums(x)
    bn = ones(Float64, length(x))
    sw = x[1:end-1] .!= x[2:end] .|> Int
    bn[2:end] = cumsum(sw) .+ 1
    return bn
end

function cumcount(x)
    return collect(Float64, 1:length(x))
end

function parse_features(s)
    return parse.(Int, split(s[2:end-1], ", "))
end

function distfromcatbound(x, y)
    return abs(7 - (x + y))
end

function fit_glmm(rawdf)
    df = @chain rawdf begin
        @subset(:condition .== "free", :stage .== "epochs")
        @select(:pid, :trialsComplete, :famInd, :features, :correct)

        @groupby(:pid)
        @transform(:blockInd = blocknums(:famInd))
        @groupby([:pid, :famInd])
        @transform(:famTrial = cumcount(:famInd))
        @groupby([:pid, :blockInd])
        @transform(:blockTrial = cumcount(:famInd))
        @transform(:correct = Float64.(:correct))

        @transform(:pid = Int.(indexin(:pid, unique(:pid))))
        @transform(:famInd = Int.(indexin(:famInd, sort(unique(:famInd)))))
        @transform(
            :trialsComplete01 = (:trialsComplete .- 1) ./ 179,
            :famTrial01 = (:famTrial .- 1) ./ 179,
            :correct = Int.(:correct)
        )
    end

    std_params = @chain df begin
        @by(:pid, 
            :m = mean(:famTrial),
            :s = std(:famTrial)
        )
    end

    df = @chain df begin
        @groupby(:pid)
        @transform(
            :trialsComplete_c = (:trialsComplete .- mean(:trialsComplete)) ./ std(:trialsComplete),
            :famTrial_c = (:famTrial .-  mean(:famTrial)) ./ std(:famTrial),
            :correct = Int.(:correct)
        )
    end

    res = fit(MixedModel, @formula(correct ~ famTrial_c + (1 + famTrial_c | pid)), df, Bernoulli())
    return df, res, std_params
end

function learning_rates(rawdf, binsize)
    df = @chain rawdf begin
        @subset(:condition .== "free", :stage .== "epochs")
        @select(:pid, :trialsComplete, :famInd, :correct)
        
        @groupby(:pid)
        @transform(:blockInd = blocknums(:famInd))
        @groupby([:pid, :famInd])
        @transform(:famTrial = cumcount(:famInd))
        @groupby([:pid, :blockInd])
        @transform(:blockTrial = cumcount(:famInd))
        @transform(:correct = Float64.(:correct))
        
        @transform(:pid = Int.(indexin(:pid, unique(:pid))))
        @transform(:famInd = Int.(indexin(:famInd, sort(unique(:famInd)))))
        @transform(
            :correct = Int.(:correct)
        )
        @select(:pid, :trialsComplete, :famTrial, :famInd, :correct)
        # @transform(:famTrialBin = ceil.(Int, :famTrial ./ binsize) .* binsize)
        @by([:famInd, :famTrial], 
            :acc = mean(:correct),
            :n = length(:correct)
        )
        @subset(:n .> 100)
    end

    fit0 = fit(LinearModel, @formula(acc ~ 1 + famTrial + famInd), df, contrasts = Dict(:famInd => EffectsCoding()))
    fit1 = fit(LinearModel, @formula(acc ~ 1 + famTrial * famInd), df, contrasts = Dict(:famInd => EffectsCoding()))
    return ftest(fit0.model, fit1.model)
end

function test_score_correlations(rawdf)
    df = @chain rawdf begin        
        @subset(:condition .== "free", :stage .== "test1")
        @transform(@byrow :featSum = sum(parse.(Int, split(strip(strip(:features, ']'), '['), ", "))))
        @subset(:featSum .!= 7)
        @transform(:famInd = Int.(indexin(:famInd, sort(unique(:famInd)))))
        @by([:pid, :famInd], :meanAcc = mean(:correct))
        unstack(:pid, :famInd, :meanAcc, renamecols = (x) -> ("fam$x"))
        @transform(
            :fam1 = Float64.(:fam1),
            :fam2 = Float64.(:fam2),
            :fam3 = Float64.(:fam3),
            :fam1r = ordinalrank(Float64.(:fam1)),
            :fam2r = ordinalrank(Float64.(:fam2)),
            :fam3r = ordinalrank(Float64.(:fam3))
        )
    end

    rs, ps = ones(Float64, 3, 3), ones(Float64, 3, 3)
    for (i, fx) in enumerate([:fam1r, :fam2r, :fam3r])
        for (j, fy) in enumerate([:fam1r, :fam2r, :fam3r])
            if i != j
                x = df[:, fx]
                y = df[:, fy]
                rs[i, j] = corspearman(x, y)
                ps[i, j] = pvalue(CorrelationTest(x, y))
            end
        end
    end
    return rs, ps
end

# # Learning outcomes
# df, res, std_params = fit_glmm(CSV.read("data/Mar2023/combined.csv", DataFrame));
# display(res)

# # Learning rates
# learning_rates(CSV.read("data/Mar2023/combined.csv", DataFrame), 10)

rs, ps = test_score_correlations(CSV.read("data/Mar2023/combined.csv", DataFrame))
display(rs)
display(ps)