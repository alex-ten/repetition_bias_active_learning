using Pkg; Pkg.activate("Blocking")

using ArviZ
using Chain
using CSV
using DataFrames, DataFramesMeta
using NCDatasets
using ProgressBars
using Turing
using CairoMakie
using StatsBase
using HypothesisTests

function converged(idata)
    res = ess_rhat(idata)
    bad_ess = any(map(ess -> ess < 100 * 4, res[1]))
    bad_rhat = any(map(rhat -> rhat > 1.01, res[2]))
    return !(bad_ess | bad_rhat)
end

function as_df(idata)
    # Get all parameter names from the posterior
    param_names = keys(idata.posterior)
    
    # Initialize an empty dictionary to store the flattened arrays
    param_arrays = Dict()
    
    # Extract and flatten each parameter's samples
    for param in param_names
        # Get the parameter values as a matrix (draws × chains)
        param_values = idata.posterior[param]
        
        # Flatten the matrix into a single vector
        param_arrays[param] = vec(param_values)
    end
    
    # Convert the dictionary to a DataFrame
    df = DataFrame(param_arrays)
    
    # Add draw and chain indicators if needed
    ndraws, nchains = size(idata.posterior[first(param_names)])
    df.draw = repeat(1:ndraws, outer=nchains)
    df.chain = repeat(1:nchains, inner=ndraws)
    
    return df
end

function calculate_hdi(x::Vector{Float64}, prob_mass::Float64=0.99)
    # Sort the data
    sorted_data = sort(x)
    n = length(sorted_data)
    
    # Calculate the number of data points to include in the HDI
    interval_size = floor(Int, prob_mass * n)
    
    # Calculate width for all possible intervals
    widths = sorted_data[(interval_size + 1):end] .- sorted_data[1:(n - interval_size)]
    
    # Find the shortest interval
    best_idx = argmin(widths)
    hdi_min = sorted_data[best_idx]
    hdi_max = sorted_data[best_idx + interval_size]
    
    return (hdi_min, hdi_max)
end

function posterior_summary(df::DataFrame; prob_mass::Float64=0.99)
    # Get parameter columns (excluding draw and chain)
    param_cols = setdiff(names(df), ["draw", "chain"])
    
    # Initialize results DataFrame
    results = DataFrame(
        parameter = String[],
        mean = Float64[],
        hdi_lower = Float64[],
        hdi_upper = Float64[]
    )
    
    # Calculate statistics for each parameter
    for param in param_cols
        values = df[:, param]
        hdi_bounds = calculate_hdi(values, prob_mass)
        
        push!(results, (
            param,
            mean(values),
            hdi_bounds[1],
            hdi_bounds[2]
        ))
    end
    
    return results
end

# Posterior summaries of the full model (mt 31) for each participant, built from the local .nc files
SUMMARY_PATH = "data/posterior_summaries_full_model2.csv"

function summarize_full_model()
    datadir = joinpath("data", "comparisons")
    piddirs = readdir(datadir)

    dfs = []
    for piddir in piddirs
        filepaths = readdir(joinpath(datadir, piddir))
        idx = findfirst(f -> split(f, "-")[2] == "31", filepaths)
        if isnothing(idx)
            @warn "Skipping $piddir — no '-31-' file"
            continue
        end
        filepath = filepaths[idx]
        idata = from_netcdf(joinpath(datadir, piddir, filepath))
        conv = converged(idata)
        df = posterior_summary(as_df(idata), prob_mass=.95)
        df.pid .= parse(Int, last(split(piddir, "-")))
        df.converged .= conv
        push!(dfs, df)
    end
    df = vcat(dfs...)
    CSV.write(SUMMARY_PATH, df)
end

function vis_individual_params()
    # The .nc files are not in the repository; use the saved summaries when available
    !isfile(SUMMARY_PATH) && summarize_full_model()
    df = CSV.read(SUMMARY_PATH, DataFrame)
    set_theme!(fontsize=20)

    fig = Figure(size=(1200, 700));
    sort!(df, :converged)
    semiblack = [(:black, 0.3) for _ in 1:121]
    for (i, pname) in enumerate(split("REP,NOV,PC,LP,RC", ","))
        ax = Axis(fig[1, i], title=replace("$pname", "RC"=>"CONF"), ylabel=i==1 ? "Participants" : "", xlabel="Estimate", yticklabelsvisible=false, yticksvisible=false, ygridvisible=false, limits=((-17, 17), nothing))
        @with @subset(df, :parameter .== pname) begin
            rangebars!(ax, 1:121, :hdi_lower, :hdi_upper, color=semiblack, direction=^(:x))
            scatter!(ax, :mean, 1:121, color=ifelse.(:converged, semiblack, ^(:white)), strokecolor=semiblack, strokewidth=1, markersize=5)
            vlines!(ax, [mean(:mean)], color=(^(:black), 1.0), linewidth=1)
            vlines!(ax, confint(:mean |> OneSampleTTest) |> collect, color=(^(:black), 1.0), linestyle=^(:dash), linewidth=1)
        end
        vlines!(ax, [0], color=:salmon)
    end
    return fig
end
fig = vis_individual_params()
save("plots/param_estimates2.pdf", fig)

df = CSV.read(SUMMARY_PATH, DataFrame)

@chain df begin
    @by(:parameter, 
        :mean = mean(:mean),
        :prop_below = mean(:hdi_upper .< 0),
        :prop_above = mean(:hdi_lower .> 0),
        :prop_around = mean(sign.(:hdi_lower) .!= sign.(:hdi_upper))
    )
end

@chain @subset(df, :converged) begin
    @by(:parameter, 
        :mean = mean(:mean),
        :lower = (:mean |> OneSampleTTest |> confint |> collect)[1],
        :upper = (:mean |> OneSampleTTest |> confint |> collect)[2],
        :prop_below = mean(:hdi_upper .< 0),
        :prop_above = mean(:hdi_lower .> 0),
        :prop_around = mean(sign.(:hdi_lower) .!= sign.(:hdi_upper))
    )
end