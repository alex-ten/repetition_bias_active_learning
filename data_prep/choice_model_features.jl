using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions

DATAPATH = "data/raw_data/combined.csv"
m = 3

include("../lib.jl")

# Read data
df = @chain CSV.read(DATAPATH, DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :correct, :confidence])
end

# Compute features
df2 = @chain df begin
    DataFrames.transform(:famInd => encode => :famCode)
    DataFrames.transform(:famCode => (ByRow(x -> dummify(x, m))) => [:c1, :c2, :c3])
    DataFrames.groupby(:pid)
    DataFrames.transform([:c1, :c2, :c3] .=> (x -> novelty(x, 3)) .=> [:nov1, :nov2, :nov3]; ungroup=false)
    DataFrames.transform([:confidence, :famCode] => ((x, cats) -> papply(x, pc, cats, m; padding=4)) => [:rc1, :rc2, :rc3]; ungroup=false)
    DataFrames.transform([:correct, :famCode] => ((x, cats) -> papply(x, pc, cats, m; padding=15)) => [:pc1, :pc2, :pc3]; ungroup=false)
    DataFrames.transform([:confidence, :famCode] => ((x, cats) -> papply(x, lp, cats, m, 10, 9; padding=15)) => [:cp1, :cp2, :cp3]; ungroup=false)
    DataFrames.transform([:correct, :famCode] => ((x, cats) -> papply(x, lp, cats, m, 10, 9; padding=15)) => [:lp1, :lp2, :lp3]; ungroup=false)
    DataFrames.transform(_, names(_, r"^(pc|lp|rc)\d+$") .=> ffill; renamecols=false) # forward-fill columns with NaNs
    DataFrames.transform(_, names(_, r"^rc\d+$") .=> (c -> replace!(c, NaN => .5)); renamecols=false)
    DataFrames.transform(_, names(_, r"^pc\d+$") .=> (c -> replace!(c, NaN => 8/15)); renamecols=false)
    DataFrames.transform(_, names(_, r"^lp\d+$") .=> (c -> replace!(c, NaN => 5/10 - 5/9)); renamecols=false)
    DataFrames.transform(_, names(_, r"^lp\d+$") .=> (c -> abs.(c)) .=> .*("abs", names(_, r"lp\d+$")))
    DataFrames.transform(_, names(_, r"^cp\d+$") .=> (c -> abs.(c)) .=> .*("abs", names(_, r"cp\d+$")))
    DataFrames.groupby([:pid, :famCode])
    DataFrames.transform(:correct => pc => :pc; ungroup=false)
    DataFrames.transform(:correct => (x -> lp(x, 10, 9)) => :lp)
    DataFrames.transform([:famCode, :abslp1, :abslp2, :abslp3] => ByRow(((i, a, b, c) -> [a, b, c][i])) => :abslp)
    DataFrames.transform([:famCode, :abscp1, :abscp2, :abscp3] => ByRow(((i, a, b, c) -> [a, b, c][i])) => :abscp)
    DataFrames.transform([:famCode, :nov1, :nov2, :nov3] => ByRow(((i, a, b, c) -> [a, b, c][i])) => :nov)
    DataFrames.transform(_, ["famCode", names(_, r"^rc\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :rcCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^pc\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :pcCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^lp\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :lpCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^abslp\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :abslpCentered)
    DataFrames.transform(_, ["famCode", names(_, r"^nov\d+$")...] => ByRow((i, a, b, c) -> ([a, b, c] .- mean([a, b, c]))[i]) => :novCentered)
    DataFrames.sort([:pid, :trialsComplete])
end

# pids in combined.csv are already anonymized integer ids (see data/private/participant_key.csv)
CSV.write("data/df_features.csv", df2)
