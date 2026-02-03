using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames
using GLM
using LogExpFunctions
using NCDatasets
using MixedModels
using JLD2
using ProgressBars
using Revise
using StatsBase
using Statistics
using Turing

includet("../lib.jl")

T = 180
Ω = [1, 2, 3]
ddf = CSV.read("data/comparisons_6000_converged_ess100x4_rhat101.csv", DataFrame)
whitelist = string.(ddf.pid, " ", ddf.name)

m = JLD2.load("simulations/predict_correct.jld2", "m")
pdirs = readdir("data/comparisons")
mfiles = readdir(joinpath("data/comparisons", pdirs[1]))
N = length(pdirs)
dfs = DataFrame[] # Empty vector for dataframes
for mfile in ProgressBar(mfiles)
    mlabel = @chain split(mfile, "-") begin
        last()
        split(".")
        first()
        replace("_" => "+")
    end
    
    for d in ProgressBar(pdirs, leave=false)
        
        _, stri, pid = split(d, "-")
        pix = parse(Int, stri)
        pid = String(pid)
        !(string(pid, " ", mlabel) in whitelist) && continue

        h = init_h(pid, pix)
        idata = from_netcdf(joinpath("data/comparisons", d, mfile)).posterior |> DataFrame
        paramsMAP = mapcols(mean, idata[:, 3:end])
        params = params_vec(names(paramsMAP), paramsMAP[1, :] |> collect)
        for t in 1:T
            x = extract_features(h)
            y = get_choice(params, x, Ω)
            fb = get_fb(y, t, m)
            push!(h, [pix, pid, mlabel, t, y, dummify(y, 3)..., fb])
        end
        push!(dfs, h)
    end
end
df = @chain vcat(dfs...) begin
    subset(:t => ByRow(!=(0)))
end
CSV.write("simulations/all_models_filtered.csv", df)