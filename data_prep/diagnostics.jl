using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using NCDatasets
using ProgressBars


datadir = joinpath("data", "comparisons")
piddirs = readdir(datadir)

df = DataFrame(pid=String[], model=String[], param=String[], diag=String[], value=Float64[])
errs1 = []
errs2 = []

# First pass to evaluate existing chains
for piddir in ProgressBar(piddirs)
    filepaths = readdir(joinpath(datadir, piddir))
    pid = @chain piddir begin
        split("-")
        last()
    end
    models = map(filepaths) do f
        mlabel = @chain f begin
            split(".")
            first()
            split("-")
            last()
            replace("_" => "+")
            Symbol()
        end
        try
            return (mlabel, from_netcdf(joinpath(datadir, piddir, f)))
        catch e
            println(e)
            push!(errs1, piddir)
            return (mlabel, nothing)
        end
    end;
    models = (; models...);
    iter = ProgressBar(keys(models); leave=false)
    for k in iter
        isnothing(models[k]) && continue
        try
            res = ess_rhat(models[k])
            set_description(iter, "model: $k")
            for i in 1:2
                for dk in keys(res[i])
                    push!(df, [pid, String(k), String(dk), i == 1 ? "ess" : "rhat", res[i][dk]])
                end
            end
        catch e
            push!(errs2, (pid=pid, model=k))
            @show k
            println(e)
            continue
        end
    end
end

# Second pass to correct errors
include("../cluster/fit_helpers.jl")
lookup = CSV.read("cluster/lookup.csv", DataFrame)
for (pid_str, mspecsymb) in errs2
    pid = parse(Int, pid_str)
    mlabel = replace(string(mspecsymb), "+" => "_")
    mt = @rsubset(lookup, :pid == pid, :label == mlabel).mt |> only
    mspec = parse.(Int, split(@rsubset(lookup, :pid == pid, :label == mlabel).code |> only, "_"))
    data = @rsubset(CSV.read("data/df_features.csv", DataFrame), :pid == pid)
    # fit_one(pid, mt, mspec, MODELS[mt], data, "data"; progress_on=true)
    m = from_netcdf("data/comparisons/pid-$pid/m-$mt-$mlabel.nc")
    res = ess_rhat(m)
    for i in 1:2
        for dk in keys(res[i])
            push!(df, [pid_str, string(mspecsymb), String(dk), i == 1 ? "ess" : "rhat", res[i][dk]])
            continue
        end
    end
end

# Manual fixes
errs1 # errs1 caught error: Opening path data/comparisons/pid-119/m-31-REP_NOV_PC_LP_RC.nc
fit_one(
    119, # pid
    31, # mspec code
    [1, 2, 3, 4, 5], # mspec
    MODELS[31], # model
    @rsubset(CSV.read("data/df_features.csv", DataFrame), :pid == 119), # data
    "data" # output directory
)
CSV.write("data/diagnostics_10k.csv", df)


# After fixing the problematic fits, there is no more errors:
# julia> errs1
# Any[]
# julia> errs2
# Any[]
