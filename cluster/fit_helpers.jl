using Turing
using LogExpFunctions
using DataFramesMeta
using Chain
using Distributions
using LinearAlgebra; BLAS.set_num_threads(1)
using ArviZ
using NCDatasets
using Logging; global_logger(ConsoleLogger(stderr, Logging.Error))


VARS = [:c, :nov, :pc, :abslp, :rc]
COEFS = [:REP, :NOV, :PC, :LP, :RC]
PRIORS = [truncated(Normal(0, 5), lower=0), truncated(Normal(0, 5), lower=0), Normal(0, 5), Normal(0, 5), Normal(0, 5)]
NCHAINS = 4
NSAMPLES = 10_000
SAMPLER = NUTS(1000, 0.65)
PARALLEL = MCMCSerial() 
MAX_ATTEMPTS = 10

function addlag(df, col, lag)
    res = df[:, :]
    res[1:end-lag, col] .= df[1+lag:end, col]
    return res
end

function rep_vars(vars::Vector{Symbol}, n::Int)
    res = []
    for v in vars, i in 1:n
        push!(res, "$(v)$(i)")
    end
    return Symbol.(res)
end

function unique_index_combinations(indices)
    unique!([sort([i for (j, i) in enumerate(indices) if (k & (1 << (j-1))) != 0])
             for k in 1:(1<<length(indices))-1])
end

function prep_data(df::DataFrame, pid, vars::Vector{Symbol})
    @chain df begin
        @rsubset(:pid == pid)
        addlag(_, :famCode, 1)
        first(_, nrow(_) - 1)
        @select(:famCode, $(vars...))
    end
end

function safe_sample(model, sampler, parallel, nsamples, nchains, max_attempts)
    attempt = 1
    while attempt <= max_attempts
        try
            return sample(model, sampler, parallel, nsamples, nchains)
        catch e
            println("  Failed to sample chains (attempt $attempt): ", sprint(showerror, e))
            attempt += 1
        end
    end
    return nothing
end


LOOKUP = CSV.read("cluster/lookup.csv", DataFrame)
MT_SPECS_MAP = Dict(
    row.mt => parse.(Int, split(row.code, "_"))
    for row in eachrow(unique(@select(LOOKUP, :mt, :code)))
)
include("models.jl")

function fit_one(pid::Integer, mt::Integer, mspec::Vector{<:Integer}, model_func::Function, data::DataFrame, output_path::String; progress_on::Bool=false)
    mlabel = join(COEFS[mspec], "_")

    mat = prep_data(data, pid, rep_vars(VARS[mspec], 3)) |> Matrix{Float64}
    Y = mat[:, 1]
    X = mat[:, 2:end]

    setprogress!(progress_on)
    model  = model_func(X, Y)

    chains = safe_sample(model, SAMPLER, PARALLEL, NSAMPLES, NCHAINS, MAX_ATTEMPTS)

    if isnothing(chains)
        return (pid=pid, mt=mt, mlabel=mlabel, status="FAILED", path=missing)
    end

    idata = from_mcmcchains(chains;
        log_likelihood = pointwise_loglikelihoods(model, chains),
        library = "Turing")

    dirpath = joinpath(output_path, "comparisons", "pid-$(pid)")
    idatapath = joinpath(dirpath, "m-$(mt)-$(mlabel).nc")
    to_netcdf(idata, idatapath)

    return (pid=pid, mt=mt, mlabel=mlabel, status="SUCCESS", path=idatapath)
end
