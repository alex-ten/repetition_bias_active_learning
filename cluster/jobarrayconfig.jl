using CSV
using DataFrames
using Chain

include(joinpath(@__DIR__, "..", "analyses", "fit_helpers.jl"))

# Model specs
vars = [:c, :nov, :pc, :abslp, :rc]
coefs = [:REP, :NOV, :PC, :LP, :RC]
mspecs_s = unique_index_combinations([1, 2, 3, 4])
mindexes_s = collect(1:length(mspecs_s))
mspecsdf = DataFrame(
    i = mindexes_s,
    label = [join(coefs[mspec], "_") for mspec in mspecs_s],
    code = [join(mspec, "_") for mspec in mspecs_s]
)
mspecs_l = filter(spec -> 5 ∈ spec, unique_index_combinations([1, 2, 3, 4, 5]))
for (j, newspec) in enumerate(mspecs_l)
    label = join(coefs[newspec], "_")
    code = join(newspec, "_")
    push!(mspecsdf, [j + length(mspecs_s), label, code])
end
mspecsdf

df = DataFrame(
    jobid = repeat(1:50, inner=96),
    subid = repeat(1:96, outer=50)
)[1:nrow(mspecsdf)*121, :]

pid = repeat(1:121, inner=31)
mt = repeat(1:31, outer=121)

length(pid) == length(mt) == nrow(df)

df.pid = pid
df.mt = mt
df.label = mspecsdf.label[df.mt]
df.code = mspecsdf.code[df.mt]


CSV.write(joinpath(@__DIR__, "lookup.csv"), df)