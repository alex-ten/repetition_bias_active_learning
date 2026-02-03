using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using Distributions

DATAPATH = "data/Mar2023/combined.csv"
m = 3

include("../lib.jl")

# Read data
df = @chain CSV.read(DATAPATH, DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :correct])
end

# Compute features
df = @chain df begin
    DataFrames.transform(:famInd => encode => :famCode)
    DataFrames.transform(:famCode => (ByRow(x -> dummify(x, m))) => [:c1, :c2, :c3])
    DataFrames.groupby(:pid)
    DataFrames.transform([:c1, :c2, :c3] .=> (x -> novelty(x, 3)) .=> [:nov1, :nov2, :nov3]; ungroup=false)
    DataFrames.transform([:correct, :famCode] => ((x, cats) -> papply(x, pc, cats, m; padding=15)) => [:pc1, :pc2, :pc3]; ungroup=false)
    DataFrames.transform([:correct, :famCode] => ((x, cats) -> papply(x, lp, cats, m, 10, 9; padding=15)) => [:lp1, :lp2, :lp3]; ungroup=false)
    DataFrames.transform(_, names(_, r"^(pc|lp)\d+$") .=> ffill; renamecols=false) # forward-fill columns with NaNs
    DataFrames.transform(_, names(_, r"^pc\d+$") .=> (c -> replace!(c, NaN => 8/15)); renamecols=false)
    DataFrames.transform(_, names(_, r"^lp\d+$") .=> (c -> replace!(c, NaN => 5/10 - 5/9)); renamecols=false)
    DataFrames.transform(_, names(_, r"^lp\d+$") .=> (c -> abs.(c)) .=> .*("abs", names(_, r"lp\d+$")))
    DataFrames.groupby([:pid, :famCode])
    DataFrames.transform(:correct => pc => :pc; ungroup=false)
    DataFrames.transform(:correct => (x -> lp(x, 10, 9)) => :lp)
    DataFrames.transform([:famCode, :abslp1, :abslp2, :abslp3] => ByRow(((i, a, b, c) -> [a, b, c][i])) => :abslp)
    DataFrames.transform([:famCode, :nov1, :nov2, :nov3] => ByRow(((i, a, b, c) -> [a, b, c][i])) => :nov)
    DataFrames.transform(_, ["famCode", names(_, r"^pc\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :pcCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^lp\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :lpCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^abslp\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :abslpCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^nov\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :novCentered)
    DataFrames.sort([:pid, :trialsComplete])
end

df[1:20, end-15:end]

# CSV.write("data/df_features.csv", df)

scatter(df.nov1, df.rep1)