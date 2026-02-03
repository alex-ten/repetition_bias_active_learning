using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames, DataFramesMeta
using NCDatasets
using ProgressBars
using Turing

CHECK_CONVERGENCE = true

function converged(idata)
    res = ess_rhat(idata)
    bad_ess = any(map(ess -> ess < 100 * 4, res[1]))
    bad_rhat = any(map(rhat -> rhat > 1.01, res[2]))
    return !(bad_ess | bad_rhat)
end

datadir = joinpath("data", "comparisons")
piddirs = readdir(datadir)

dfs = []
for (pidindex, piddir) in enumerate(piddirs[1:end])
    filepaths = readdir(joinpath(datadir, piddir))
    progbar = ProgressBar(total=length(filepaths))

    models = map(filepaths) do f
        mlabel = @chain f begin
            split(".")
            first()
            split("-")
            last()
            replace("_" => "+")
            Symbol()
        end
        set_description(progbar, "Reading NetCDF files ($pidindex): ")
        update(progbar)
        idata = from_netcdf(joinpath(datadir, piddir, f))
        converged_res = converged(idata)
        return converged_res ? (mlabel, idata) : (mlabel, nothing)
    end;
    models = filter(p -> !(p[2] |> isnothing), models)
    models = (; models...);

    # # Perform comparisons
    psis_df = compare(models; elpd_method=loo) |> DataFrame
    waic_df = compare(models; elpd_method=waic) |> DataFrame

    # # # Store data
    psis_df[!, :method] .= "psis"
    waic_df[!, :method] .= "waic"
    merged = vcat(psis_df, waic_df)
    merged[!, :pid] .= last(split(piddir, "-"))
    push!(dfs, merged)
end

df = vcat(dfs...)
CSV.write("data/comparisons_6000_converged_ess100x4_rhat101.csv", df)