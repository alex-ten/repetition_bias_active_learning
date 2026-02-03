using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using GLM
using CairoMakie
using JLD2
using MixedModels
using Random
using RCall
using Statistics
using StatsBase

include("../lib.jl")


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

function plot_data(data::Matrix)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="blockTrial", ylabel="famTrial")
    for cat in 0:1
        subdata = data[data[:, 3] .== cat, :]
        jit = rand(Uniform(-0.3, 0.3), size(subdata)...)
        scatter!(ax, subdata[:, 1] + jit[:, 1], subdata[:, 2] + jit[:, 2], color=ifelse(cat |> Bool, :green, :red), markersize=5)
    end
    fig
end

function split_df(df, prop_train=0.7)
    # Create a shuffled index
    shuffled_indices = shuffle(1:nrow(df))
    
    # Calculate the split point
    split_point = Int(floor(nrow(df) * prop_train))
    
    # Split the dataframe using the shuffled indices
    train = df[shuffled_indices[1:split_point], :]
    test = df[shuffled_indices[split_point+1:end], :]
    
    return train, test
end

# Read data
df = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :features, :correct])
    DataFrames.transform(:features => ByRow(parse_features) => [:f1, :f2])
    select(Not([:features]))
    DataFrames.transform([:f1, :f2] => ByRow(distfromcatbound) => :dist)
    select(Not([:f1, :f2]))
end

# Compute features
df2 = @chain df begin
    @groupby(:pid) # needed in case a person does try do all tasks
    @transform(:famInd = encode(:famInd))
    @groupby(:pid)
    @transform(
        :blockInd = blocknums(:famInd)
    )
    @groupby([:pid, :famInd])
    @transform(
        :famTrial = cumcount(:famInd)
    )
    @groupby([:pid, :blockInd])
    @transform(:blockTrial = cumcount(:famInd))
    @transform(:correct = Float64.(:correct))
    @transform(:famInd = string.(:famInd))
    @groupby([:pid, :famInd])
    @transform(
        :trialsPerFam = length(:famInd),
        :corrPerTrialFam = cumsum(:correct)
    )
end

df3 = @chain df2 begin
    @groupby([:pid, :famInd])
    @combine(
        :acc = mean(:correct),
        :nTrials = length(:correct)
    )
    @groupby(:famInd)
    @combine(
        :acc = mean(:acc),
        :nTrials = mean(:nTrials)
    )
    sort(:famInd)
end

# Anova on effect of task on learning
function test_performance(df, n)
    rdf = @chain df begin
        @subset(:famTrial .< n)
        @groupby([:pid, :famInd])
        @combine(
            :acc = mean(:correct)
        )
    end
    R"""
        library(lme4)
        library(lmerTest)

        model <- lmer(acc ~ famInd + (1 | pid), data = $(rdf))
        print(model)
        cat("\n\n")
        anova(model)
    """
end
test_performance(df2, 30)
test_performance(df2, 50)

# Effect of trial on accuracy
function test_learning(df)
    m = fit(MixedModel, @formula(correct ~ trialsComplete + (1 + trialsComplete | famInd)), df, Bernoulli())
end
test_learning(df2)

include("../vislib.jl")
function get_dfsum()
    df = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
        @subset(:condition .== "free")
        @subset(:stage .== "epochs")
        select([:pid, :trialsComplete, :famInd, :correct])
        rename(:famInd => "famCode")
        groupby(:pid)
        addcols()
        # add_pid_numbers!
        # @aside CSV.write("visualization/plot_sum_df_hum.csv", _)
    end
    dfsum = groupby(df, :pid) |> summarize
    return dfsum
end
dfsum = get_dfsum()

# The effect of blocking on accuracy
dftest = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
    @subset(@byrow sum(parse.(Int, split(chop(:features, head=1), ','))) != 7)
    subset(:condition => ByRow(==("free")))
    @subset(:stage .== "test1" .|| :stage .== "test2")
    @groupby(:pid)
    @transform(:famInd = string.(encode(:famInd)))
    @by([:pid, :famInd, :stage],
        :testacc = mean(:correct)
    )
end

lengths = [nrow(@chain @subset(temp, :pid .== pid, :stage .== "test1") @subset(@byrow sum(parse.(Int, split(chop(:features, head=1), ','))) != 7)) for pid in unique(dftest.pid)]
hist(lengths, bins=60:1:90)

dfjoined = @chain innerjoin(df2, dftest, on=[:pid, :famInd]) begin
    @by([:pid, :stage],
        :acc = mean(:correct),
        :testacc = mean(:testacc),
    )
    innerjoin(dfsum, on=:pid)
    # @subset(:numSwitches .> 5)
    # @transform(:blockSizeVar = log.(:blockSizeVar))
    @transform(
        :numSwitches_c = (:numSwitches .- mean(:numSwitches)) ./ std(:numSwitches),
        :blockSizeVar_c = (:blockSizeVar .- mean(:blockSizeVar)) ./ std(:blockSizeVar)
    )
end

function testacc(df)
    R"""
        library(lme4)
        library(lmerTest)
        library(dplyr)
        library(tibble)

        df1 <-  dplyr::filter($(df), stage=="test1") |> as_tibble()
        print(head(df1))
        model1 <- lm(testacc ~ numSwitches_c * blockSizeVar_c, data = df1)
        print(summary(model1))

        df2 <-  dplyr::filter($(df), stage=="test2") |> as_tibble()
        print(head(df2))
        model2 <- lm(testacc ~ numSwitches_c * blockSizeVar_c, data = df2)
        print(summary(model2))
    """

    return nothing
end

testacc(dfjoined)

# Dropouts
@chain @subset(dfjoined, :stage .== "test1") @select(:pid) unique
@chain @subset(dfjoined, :stage .== "test2") @select(:pid) unique
