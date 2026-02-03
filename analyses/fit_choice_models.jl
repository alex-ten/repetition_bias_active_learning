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
function define_model(specs, priors)
    model_expr = quote
        @model function clogit(x, y)
            nrows, ncols = size(x)    # dimensionality of input matrix
            d = ncols ÷ 3          # number of feature groups (e.g., pc1, pc2, pc3 is one group) 
            g = ncols ÷ d          # size of each feature group (e.g., pc1, pc2, pc3 is of size 3)
            length(y) == nrows || throw(DimensionMismatch("number of observations in `z` and `y` is not equal"))

            # Priors of coefficients
            τ ~ Exponential(5)
            $([:($spec ~ $prior) for (spec, prior) in zip(specs, priors)]...)

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

function safe_sample(model, sampler, parallel, nsamples, nchains, max_attempts)
    let attempt = 1
        while attempt <= max_attempts
            try
                # Sample from posterior
                return sample(model, sampler, parallel, nsamples, nchains)
            catch e
                println("  Failed to sample chains. Trying again")
                attempt += 1
            end
        end
    end
    return nothing
end

include("fit_helpers.jl")

NCHAINS = 3
NSAMPLES = 6000
SAMPLER = NUTS(800, 0.65)
PARALLEL = MCMCThreads()
MAX_ATTEMPTS = 3

@time begin
    # Get features dataset (main data to fit)
    df = CSV.read("data/df_features.csv", DataFrame)
    pids = unique(df.pid)
    npids = length(pids)

    # Model specs
    vars = [:c, :nov, :pc, :abslp]
    coefs = [:REP, :NOV, :PC, :LP]
    priors = [truncated(Normal(0, 5), lower=0), truncated(Normal(0, 5), lower=0), Normal(0, 5), Normal(0, 5)]

    # Combos
    mspecs = unique_index_combinations([1, 2, 3, 4])
    
    # Problems log df
    problems = DataFrame(pindex=Int[], pid=String[], mspec=String[])
    for (pindex, pid) in enumerate(pids[1:end])
        # pindex < 21 && continue
        println("PID $(pindex)/$(npids) [$(pid)]")

        # Create directories to save data
        dirpath = joinpath("data", "comparisons_$(NSAMPLES)", "pid-$(pindex)-$(pid)")
        !isdir(dirpath) && mkpath(dirpath)

        for (mindex, mspec) in enumerate(mspecs[1:end])
            mlabel = join(coefs[mspec], "_")
            print("  $(mindex). $(mlabel)\n")

            # Extract data for given model and format for Turing
            data = prep_data(df, pid, rep_vars(vars[mspec], 3)) |> Matrix{Float64}
            Y = data[:, 1]
            X = data[:, 2:end]

            # Perform inference
            setprogress!(false)
            clogit = define_model(coefs[mspec], priors[mspec])
            model = clogit(X, Y)
        
            sample_chains = safe_sample(model, SAMPLER, PARALLEL, NSAMPLES, NCHAINS, MAX_ATTEMPTS)
            
            if isnothing(sample_chains)
                push!(problems, [pindex, pid, mlabel])
                println(" ❗ Something went wrong!")
            else
                # Create inference data object
                idata = from_mcmcchains(sample_chains;
                    log_likelihood = pointwise_loglikelihoods(model, sample_chains),
                    library = "Turing"
                )
                
                # Save to .nc file
                idatapath = joinpath(dirpath, "m-$(mindex)-$(mlabel).nc")
                to_netcdf(idata, idatapath)
                println(" ✅")
            end
        end
        break
    end
end

# CSV.write("data/fit_errors.csv", problems)

using HypothesisTests
pids = unique(df.pid)
rs = []
ps = []
for pid in pids
    m = prep_data(df, pid, rep_vars(vars[[1, 2]], 3)) |> Matrix{Float64}
    ct1 = HypothesisTests.CorrelationTest(m[:, 2], m[:, 5])
    ct2 = HypothesisTests.CorrelationTest(m[:, 3], m[:, 6])
    ct3 = HypothesisTests.CorrelationTest(m[:, 4], m[:, 7])
    push!(rs, [res.r for res in [ct1, ct2, ct3]])
    push!(ps, [pvalue(res) for res in [ct1, ct2, ct3]])
end
rs = hcat(rs...) |> transpose |> collect
ps = hcat(ps...) |> transpose |> collect
fig = Figure(size=(1000, 800));
for i in 1:3
    ax = Axis(fig[i, 1], xlabel="Cor. coef", title="Family $i", limits=((-.8, .1), (0, nothing)))
    x = rs[:, i]
    hist!(ax, x[.!isnan.(x)], bins=-.8:.02:.1)
    vlines!(ax, 0, linestyle=:dash, color=:black)
end
fig