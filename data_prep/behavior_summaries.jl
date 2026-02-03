using Pkg; Pkg.activate("Blocking")
using CSV
using Chain
using DataFrames
using JLD2
using Revise
using StatsBase

includet("../vislib.jl")

# Random simulated data
@chain CSV.read("simulations/df_rand.csv", DataFrame) begin
    groupby(:pid)
    addcols()
    @aside CSV.write("visualization/plot_df_rand.csv", _)
    groupby(:pid)
    summarize()
    @aside CSV.write("visualization/plot_sum_df_rand.csv", _)
end

# Human data
@chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :correct])
    DataFrames.transform(:correct => ByRow(Int) => :correct)
    rename(:famInd => "famCode")
    groupby(:pid)
    addcols()
    @aside CSV.write("visualization/plot_df_hum.csv", _)
    groupby(:pid)
    summarize()
    @aside CSV.write("visualization/plot_sum_df_hum.csv", _)
end

# Model-based simulated data
@chain CSV.read("simulations/all_models_filtered.csv", DataFrame) begin
    groupby([:model, :pid])
    addcols()
    @aside CSV.write("visualization/plot_df_sim.csv", _)
    groupby([:model, :pid])
    summarize()
    @aside CSV.write("visualization/plot_sum_df_sim.csv", _)
end


using Distributions, Statistics, StatsBase


function f()
    X = rand(Distributions.Categorical(ones(3)./3), 180)
    switch = X[1:length(X)-1] .!= X[2:length(X)]
    blocks = cumsum(switch)
    lbsv = blocks |> countmap |> values |> var
    return [sum(switch), lbsv]
end

m = Matrix{Float64}(undef, 100, 2)
for i in 1:size(m, 1)
    m[i, :] = f()
end
mean(m, dims=1)

@chain CSV.read("simulations/df_rand.csv", DataFrame) begin
    groupby(:pid)
    addcols()
    @aside CSV.write("visualization/plot_df_rand.csv", _)
    groupby(:pid)
    summarize()
    @aside CSV.write("visualization/plot_sum_df_rand.csv", _)
end