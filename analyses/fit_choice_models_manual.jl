using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames
using DataFramesMeta
using Distributions
using LinearAlgebra
using LogExpFunctions
using Logging; global_logger(ConsoleLogger(stderr, Logging.Error))
using NCDatasets
using Turing

# Refit the models that are missing for a single participant, using the cluster code.
# Run from the repository root.
include("../cluster/fit_helpers.jl")

# Local overrides (cluster uses MCMCSerial with one job per model)
PARALLEL = MCMCThreads()

pid = 121
df = CSV.read("data/df_features.csv", DataFrame)
dirpath = joinpath("data", "comparisons", "pid-$(pid)")
!isdir(dirpath) && mkpath(dirpath)
files = readdir(dirpath)
for mt in unique(LOOKUP.mt)
    mspec = MT_SPECS_MAP[mt]
    mlabel = join(COEFS[mspec], "_")
    "m-$mt-$mlabel.nc" in files && continue
    println("fit_one($pid, $mt, $mspec, $(MODELS[mt]), @rsubset(df, :pid == $pid), \"data\")")
    fit_one(pid, mt, mspec, MODELS[mt], @rsubset(df, :pid == pid), "data")
end
