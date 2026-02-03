using Pkg; Pkg.activate("Blocking")
using ArviZ
using GLMakie
using Chain
using CSV
using DataFrames, DataFramesMeta
using Distributions
using NCDatasets
using ProgressBars


datadir = joinpath("data", "comparisons")
piddirs = readdir(datadir)

df = DataFrame(pid=String[], model=String[], param=String[], diag=String[], value=Float64[])
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
        return (mlabel, from_netcdf(joinpath(datadir, piddir, f)))
    end;
    models = (; models...);
    for k in ProgressBar(keys(models); leave=false)
        res = ess_rhat(models[k])
        for i in 1:2
            for dk in keys(res[i])
                push!(df, [pid, String(k), String(dk), i == 1 ? "ess" : "rhat", res[i][dk]])
            end
        end
    end
end
CSV.write("data/diagnostics_6000.csv", df)
