using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using CairoMakie
using DataFrames, DataFramesMeta
using Distributions
using MixedModels, GLM

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

function accuracy_difficulty_plot(df)
    cor = round(mean(df.correct), digits=2) 
    inc = round(1-cor, digits=2)

    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1], ylabel="Distance from class boundary (difficulty)", xlabel="Accuracy", xticks=([0, 1], ["Incorrect ($(inc * 100)%)", "Correct ($(cor * 100)%)"]))
    scatter!(ax, Int.(df.correct) .+ rand(nrow(df)) ./ 2 .- 0.5/2, df.dist .+ rand(nrow(df)) ./ 2 .- 1/4, markersize=5, alpha=.3)
    fig
end
# accuracy_difficulty_plot(df)

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

function get_test_data(rawdf)
    df = @chain rawdf begin        
        @subset(:condition .== "free", :stage .== "test1")
        @transform(@byrow :featSum = sum(parse.(Int, split(strip(strip(:features, ']'), '['), ", "))))
        @subset(:featSum .!= 7)
        @transform(:famInd = Int.(indexin(:famInd, sort(unique(:famInd)))))
        @by([:pid, :famInd], :meanAcc = mean(:correct))
    end
end

function learning_plot(DF, res, std_params, df2; binsize=10)
    fig = Figure(size=(1000, 400))
    ax1 = Axis(fig[1, 1], title="Aggregated raw data", xlabel="Trials per family (bins of 10)", ylabel="Observed mean accuracy", limits=((0, 181), (-0.05, 1.05)))
    ax2 = Axis(fig[1, 2], title="Fitted model", xlabel="Trials per family", ylabel="Modeled expected accuracy", limits=((0, 181), (-0.05, 1.05)))
    ax3ylabels = replace.([@sprintf("(%.2f, %.2f]", l, l + 0.1) for l in 0:0.1:0.9], r"\b0\." => ".")
    ax3 = Axis(fig[1, 3], title="Immediate test accuracy", xlabel="Stimulus family", ylabel="Accuracy (bin)", xticks=([1, 2, 3], string.([1, 2, 3])), yticks=(collect(1:10), ax3ylabels))
    
    df = copy(DF)
    df.correctPred = predict(res)

    # Raw data
    pids = unique(df.pid)
    for pid in pids
        sdf = @chain @subset(df, :pid .== pid) begin
            @aside lines!(ax2, _.famTrial, _.correctPred, color=(:gray, .1))
            @transform(:bin = ceil.(Int, :famTrial ./ binsize) .* binsize)
            @by(:bin,
                :sumAcc = sum(:correct),
                :nFams = length(:correct)
            )
            # @subset(:nFams .>= binsize * 3)
        end
        lines!(ax1, sdf.bin, sdf.sumAcc ./ sdf.nFams, color=(:gray, .1))
    end

    sumdf = @chain df begin
        @transform(:bin = ceil.(Int, :famTrial ./ binsize) .* binsize)
        @by(:bin,
            :sumAcc = sum(:correct),
            :nFams = length(:correct)
        )
    end

    scatterlines!(ax1, sumdf.bin, sumdf.sumAcc ./ sumdf.nFams, color=:red)

    m = mean(std_params.m)
    s = mean(std_params.s)

    famTrial_range = 1:180
    newdf = DataFrame(
        famTrial   = collect(famTrial_range),
        famTrial_c = (collect(famTrial_range) .- m) ./ s,
        pid        = fill(-1, 180),
        correct    = fill(0, 180)
    )
    fixedPred = predict(res, newdf; new_re_levels=:population)
    lines!(ax2, famTrial_range, fixedPred, color=:red)

    # Test histograms
    dftest = @chain df2 begin 
        @aside binsize = 10
        @transform(:bin = ceil.(Int, (:meanAcc .* 100) ./ binsize) .* binsize)
        unstack(:pid, :famInd, :bin, renamecols = (x) -> ("fam$x"))
        @transform(
            :fam1 = Float64.(:fam1),
            :fam2 = Float64.(:fam2),
            :fam3 = Float64.(:fam3)
        )
    end
    bins = 10:10:100
    fams = [:fam1, :fam2, :fam3]
    props = [sum(dftest[!, fam] .== bin) for bin in bins, fam in fams] ./ 121

    hm = heatmap!(ax3, props', colormap=:batlow)
    Colorbar(fig[1, 4], hm, label="Proportion (N = 121)")

    colsize!(fig.layout, 1, Relative(0.325))
    colsize!(fig.layout, 2, Relative(0.325))
    colsize!(fig.layout, 3, Relative(0.20))
    colsize!(fig.layout, 4, Relative(0.05))

    Label(fig[1, 1, TopLeft()], "A", fontsize=18, font=:bold, padding=(0, 0, 5, 0))
    Label(fig[1, 2, TopLeft()], "B", fontsize=18, font=:bold, padding=(0, 0, 5, 0))
    Label(fig[1, 3, TopLeft()], "C", fontsize=18, font=:bold, padding=(0, 0, 5, 0))

    fig
end

# Fit learning model
df, res, std_params = fit_glmm(CSV.read("data/Mar2023/combined.csv", DataFrame));
df2 = get_test_data(CSV.read("data/Mar2023/combined.csv", DataFrame))

fig = learning_plot(df, res, std_params, df2; binsize=10);
save("plots/learning.pdf", fig)