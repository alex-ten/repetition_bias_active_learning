using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using LogExpFunctions
using Logging; global_logger(ConsoleLogger(stderr, Logging.Error))
using MLDataUtils: shuffleobs, splitobs, rescale!
using NCDatasets
using ProgressBars
using Turing

# Bayesian conditional (multinomial) logistic regression
function define_model(specs)
    model_expr = quote
        @model function clogit(x, y)
            nrows, ncols = size(x)    # dimensionality of input matrix
            d = ncols ÷ 3          # number of feature groups (e.g., pc1, pc2, pc3 is one group) 
            g = ncols ÷ d          # size of each feature group (e.g., pc1, pc2, pc3 is of size 3)
            length(y) == nrows || throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

            # Priors of coefficients
            τ ~ Exponential(5)
            $([:($spec ~ Normal(0, 5)) for spec in specs]...)

            # Coefficient vector
            β = [$([:($spec) for spec in specs]...)]

            # Compute the likelihood of the observations
            z = +([
                β[i] .* x[:, k:k+g-1] 
                for (i, k) in enumerate(1:g:ncols)
            ]...)

            # Subtract maximum for stability
            z .-= maximum(z, dims=2)

            # Scale by temperature
            z ./= τ

            for i in eachindex(y)
                p = softmax(z[i, :])
                y[i] ~ Turing.Categorical(p)
            end
        end
    end
    eval(model_expr)
    return clogit
end

function safe_sample(model, sampler, parallel, nsamples, nchains)
    try
        # Sample from posterior
        return sample(model, sampler, parallel, nsamples, nchains)
    catch e
        return nothing
    end
end

include("fit_helpers.jl")
NCHAINS = 3
NSAMPLES = 500
SAMPLER = NUTS(500, 0.65)
PARALLEL = MCMCThreads()

errors = CSV.read("data/fit_errors.csv", DataFrame)

# Fit
@time begin
    # Base.run(`clear`)

    # Get features dataset (main data to fit)
    df = CSV.read("data/df_features.csv", DataFrame)
    pids = unique(df.pid)
    npids = length(pids)

    # Model specs
    vars = [:c, :nov, :pc, :abslp]
    coefs = [:REP, :NOV, :PC, :LP]

    # Combos
    mspecs = unique_index_combinations([1, 2, 3, 4])
    
    for (pindex, pid) in enumerate(pids[1:end])
        !(pid in errors.pid) && continue

        println("PID $(pindex)/$(npids) [$(pid)]")

        # Create directories to save data
        dirpath = joinpath("data", "comparisons", "pid-$(pindex)-$(pid)")
        !isdir(dirpath) && mkpath(dirpath)

        for (mindex, mspec) in enumerate(mspecs[1:end])
            mlabel = join(coefs[mspec], "_")
            !(mlabel in errors[errors.pid .== pid, :mspec]) && continue
            print("  $(mindex). $(mlabel)")

            # Extract data for given model and format for Turing
            data = prep_data(df, pid, rep_vars(vars[mspec], NCHAINS)) |> Matrix{Float64}
            Y = data[:, 1]
            X = data[:, 2:end]

            # Perform inference
            setprogress!(false)
            clogit = define_model(coefs[mspec])
            model = clogit(X, Y)
            sample_chains = safe_sample(model, SAMPLER, PARALLEL, NSAMPLES, NCHAINS)
            
            if isnothing(sample_chains)
                println(" ❗ Something went wrong!")
            else
                # Create inference data object
                idata = from_mcmcchains(sample_chains;
                    log_likelihood = cat(collect(values(pointwise_loglikelihoods(model, sample_chains)))..., dims=3),
                    library = "Turing"
                )
                
                # Save to .nc file
                idatapath = joinpath(dirpath, "m-$(mindex)-$(mlabel).nc")
                to_netcdf(idata, idatapath)
                println(" ✅")
            end
        end
    end
end