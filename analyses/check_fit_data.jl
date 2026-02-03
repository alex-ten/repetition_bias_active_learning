using Pkg; Pkg.activate("Blocking")
using ArviZ
using Chain
using CSV
using DataFrames
using Distributions
using Turing

function unique_index_combinations(indices)
    unique!([
        sort([i for (j,i) in enumerate(indices) if (k & (1 << (j-1))) != 0])
        for k in 1:(1<<length(indices))-1
    ])
end

# Check for missing data
datadir = joinpath("data", "comparisons")
pids = CSV.read("data/df_features.csv", DataFrame; select=[:pid]) |> unique 
piddirs = readdir(datadir)
pids_present = getindex.(split.(piddirs, '-'), 3)

# Check if all participants were attempted
if nrow(pids) > length(pids_present)
    println("There are less pid directories than pids")
else
    println("The number of pid directories matches the number of pids")
end

# Check if there were problems while processing each participant
coefs = [:REP, :NOV, :PC, :LP] .|> String
mspecs = [
    join(coefs[inds], "_") 
    for inds in unique_index_combinations([1, 2, 3, 4])
]

# Combos
mspecs_absent = DataFrame(pid=String[], mspec=[])
for piddir in piddirs
    mspecs_present = map( joinpath(datadir, piddir) |> readdir ) do filename
        @chain filename begin
            split("-")
            last()
            split(".")
            first()
        end
    end
    if length(mspecs_present) < length(mspecs)
        for mspec in mspecs
            !(mspec in mspecs_present) && push!(mspecs_absent, [last(split(piddir, "-")), mspec])
        end
    end
end

if isempty(mspecs_absent)
    println("All participants have all models.")
else
    savepath = "data/fit_errors.csv"
    CSV.write(savepath, mspecs_absent)
    println("Some participants have missing data. Info saved to $savepath.")
end
