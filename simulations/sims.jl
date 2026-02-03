using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using LogExpFunctions
using GLM
using MixedModels
using JLD2
using StatsBase
using Statistics
using Turing

include("../lib.jl")

m = JLD2.load("predict_correct.jld2", "m")
function get_fb(y, i, m)
    p = predict(m, DataFrame(correct=[0.0], trialsComplete=[i], famInd=["$y"]))[1]
    return Int(rand() < p)
end


function sample_params(mfile::String)
    path = rand(readdir(datadir))
    chain = load(datadir * "/" * path)["chain"]
    d = Dict(
        :τ => 1.0,
        :wREP => 0.0,
        :wNOV => 0.0,
        :wPC => 0.0,
        :wLP => 0.0
    )
    for k in d |> keys
        if k in chain.name_map.parameters
            d[k] = mean(chain)[k, :mean]
        end
    end
    return [d[:wREP], d[:wNOV], d[:wPC], d[:wLP], d[:τ]]
end

pdirs = readdir("data/comparisons")
mfiles = readdir(joinpath("data/comparisons", pdirs[1]))
N = length(dirs)
for mfile in mfiles
    T = 180
    Ω = [1, 2, 3]
    df = DataFrame[] # Empty vector for dataframes
    for d in pdirs
        _, stri, pid = split(d, "-")
        h = init_h(String(pid), parse(Int, stri))
        θ = sample_params(mfile)
        # for t in 1:T
        #     x = extract_features(h)
        #     y = get_choice(θ, x, Ω)
        #     fb = get_fb(y, t, m)
        #     push!(h, [i, t, y, dummify(y, 3)..., fb])
        # end
        # push!(df, h)
    end
    # df = @chain vcat(df...) begin
    #     subset(:t => ByRow(!=(0)))
    # end
    # CSV.write("simulations/df_$(s)_modfb.csv", df)
end