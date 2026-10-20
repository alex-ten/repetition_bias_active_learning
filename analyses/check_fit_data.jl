using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames

# Expected fits: one row per (pid, model type) in the cluster lookup table
lookup = CSV.read("cluster/lookup.csv", DataFrame)

# Check for missing data
datadir = joinpath("data", "comparisons")
piddirs = readdir(datadir)
pids_present = parse.(Int, getindex.(split.(piddirs, '-'), 2))

# Check if all participants were attempted
if length(unique(lookup.pid)) > length(pids_present)
    println("There are less pid directories than pids")
else
    println("The number of pid directories matches the number of pids")
end

# Check if there were problems while processing each participant
mspecs_absent = DataFrame(pid=Int[], mt=Int[], label=String[])
for row in eachrow(lookup)
    filepath = joinpath(datadir, "pid-$(row.pid)", "m-$(row.mt)-$(row.label).nc")
    !isfile(filepath) && push!(mspecs_absent, [row.pid, row.mt, row.label])
end

if isempty(mspecs_absent)
    println("All participants have all models.")
else
    savepath = "data/fit_errors.csv"
    CSV.write(savepath, mspecs_absent)
    println("Some participants have missing data. Info saved to $savepath.")
end
