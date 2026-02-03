using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames, DataFramesMeta
using StatsBase


df = CSV.read("data/diagnostics_6000.csv", DataFrame)
df = unstack(df, [:pid, :model, :param], :diag, :value)
df.bad_ess = df.ess .< 4*100;
df.bad_rhat = df.rhat .> 1.01;
df.bad_both = df.bad_ess .&& df.bad_rhat;
@chain df begin
    groupby([:pid, :model])
    combine(:bad_ess => any, :bad_rhat => any)
    @aside display(_)
    groupby(:model)
    combine(:bad_ess_any => sum, :bad_rhat_any => sum)
end

