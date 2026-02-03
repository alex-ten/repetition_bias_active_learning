using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using StatsBase


function groupcounts(df, col)
    return combine(groupby(df, col), col => length)
end

# Main dataset
df = @chain CSV.read("data/Mar2023/combined.csv", DataFrame) begin
    subset(:condition => ByRow(==("free")))
end

# Demographics
demo = @chain CSV.read("data/Mar2023/free/free_demo.csv", DataFrame) begin
    subset("Participant id" => ByRow(in(unique(df.pid))))
end
names(demo)

# Get total number of participants
N = length(unique(demo[:, "Participant id"]))

# Proportion female
sum(demo.Sex .== "Female") ./ N

# Age
age = parse.(Int, demo.Age)
extrema(age)
median(age)

# Country
country_df = groupcounts(demo, "Country of residence")
country_df[!, :prop] = country_df[:, 2] ./ N

# Language
lang_df = groupcounts(demo, "Fluent languages")
lang_df[!, :prop] = lang_df[:, 2] ./ N

# Duration of data collection
extrema(demo[:, "Completed at"])

