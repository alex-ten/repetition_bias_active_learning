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

# Local counterpart of the cluster job array. Model specs, priors, sampler settings and
# model definitions all come from the cluster code so local fits match cluster fits.
# Run from the repository root: `julia --threads=4 analyses/fit_choice_models.jl`
include("../cluster/fit_helpers.jl")

# Local overrides (cluster uses MCMCSerial with one job per model)
PARALLEL = MCMCThreads()

# Subsets to fit (set to e.g. [1] and [1, 2] for a quick test)
PIDS = unique(LOOKUP.pid)
MTS = unique(LOOKUP.mt)

@time begin
    # Get features dataset (main data to fit)
    df = CSV.read("data/df_features.csv", DataFrame)
    npids = length(PIDS)

    # Problems log df
    problems = DataFrame(pid=Int[], mt=Int[], label=String[])
    for pid in PIDS
        println("PID $(pid)/$(npids)")

        # Create directories to save data
        dirpath = joinpath("data", "comparisons", "pid-$(pid)")
        !isdir(dirpath) && mkpath(dirpath)

        for mt in MTS
            mspec = MT_SPECS_MAP[mt]
            print("  $(mt). $(join(COEFS[mspec], "_"))")

            res = fit_one(pid, mt, mspec, MODELS[mt], @rsubset(df, :pid == pid), "data")

            if res.status == "FAILED"
                push!(problems, [pid, mt, res.mlabel])
                println(" ❗ Something went wrong!")
            else
                println(" ✅")
            end
        end
    end
end

# CSV.write("data/fit_errors.csv", problems)

# using HypothesisTests
# pids = unique(df.pid)
# rs = []
# ps = []
# for pid in pids
#     m = prep_data(df, pid, rep_vars(vars[[1, 2]], 3)) |> Matrix{Float64}
#     ct1 = HypothesisTests.CorrelationTest(m[:, 2], m[:, 5])
#     ct2 = HypothesisTests.CorrelationTest(m[:, 3], m[:, 6])
#     ct3 = HypothesisTests.CorrelationTest(m[:, 4], m[:, 7])
#     push!(rs, [res.r for res in [ct1, ct2, ct3]])
#     push!(ps, [pvalue(res) for res in [ct1, ct2, ct3]])
# end
# rs = hcat(rs...) |> transpose |> collect
# ps = hcat(ps...) |> transpose |> collect
# fig = Figure(size=(1000, 800));
# for i in 1:3
#     ax = Axis(fig[i, 1], xlabel="Cor. coef", title="Family $i", limits=((-.8, .1), (0, nothing)))
#     x = rs[:, i]
#     hist!(ax, x[.!isnan.(x)], bins=-.8:.02:.1)
#     vlines!(ax, 0, linestyle=:dash, color=:black)
# end
# fig