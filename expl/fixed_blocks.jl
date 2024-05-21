using Chain
using CSV
using DataFrames
using Distributions
using GLMakie

function myfilter(condition, stage)::Bool
    only_free = condition == "free"
    only_epochs = stage == "epochs"
    only_free && only_epochs
end

function switch(famInd)
    res = zeros(Bool, length(famInd))
    res[2:end] = famInd[2:end] .!= famInd[1:end-1]
    return res
end

function rolling_mean(x; x0=0.0, α=0.01)
    res = []
    m = x0
    for (i, el) in enumerate(x)
        m += α * (el - m)
        push!(res, m)
    end
    return res
end

begin
    blockSizes = collect(6:20)
    df = @chain begin
        CSV.read("data/Mar2023/combined.csv", DataFrame)
        subset([:condition, :stage] => ByRow(myfilter))
        select([:pid, :trialsComplete, :famInd, :confidence, :correct])
        groupby(:pid)
        transform(:famInd => switch => :switch)
        groupby(:pid)
        transform(:switch => (x -> cumsum(x) .+ 1) => :blockNum)
        groupby([:pid, :blockNum])
        transform(:blockNum => (x -> fill(length(x), length(x))) => :blockSize)
    end

    fig = Figure(size=(1500, 900))
    maxrows = 3
    grid = reshape(1:15, 3, :)
    for (i, blockSize) in enumerate(blockSizes)
        x = collect(1:blockSize)
        ix = indexin(i, grid)[1]
        ax = Axis(fig[ix[1], ix[2]], limits=((1, blockSize), nothing), xticks=(x, ["$s" for s in x]), xlabel="Trial number", ylabel="y")
        for g in groupby(subset(df, :blockSize => ByRow(==(blockSize))), [:pid, :blockNum])
            dconf = g.confidence[2:end] - g.confidence[1:end-1]
            insert!(dconf, 1, NaN)
            dconfabs = abs.(dconf)
            conf = g.confidence
            corr = rolling_mean(g.correct; x0=0.5, α=.01) .+ ((rand()-0.5)/100)
            y = corr
            lines!(ax, x, y, linewidth=2, alpha=.4)
            # scatter!(ax, x, y, alpha=.5)
        end
    end

    display(fig)
    save("img/n-blocks_corrrm.png", fig)
end

