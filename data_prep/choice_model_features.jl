using Pkg; Pkg.activate("Blocking")
using Chain
using CSV
using DataFrames
using Distributions

DATAPATH = "data/Mar2023/combined.csv"
K = 3

"""
Encodes a categorical column of arbitrary encoding into a consistent numerical encoding. 
This ensures that categories are properly encoded. I.e., if there are K categories, 
exactly K integers from 1 to K will be assigned to each category respectively.
"""
function encode(x)
    y = x[:]
    u = sort(unique(y))
    for i in eachindex(u)
        y[x .== u[i]] .= i
    end
    return y
end

"""
Expands a categorical encoding into a dummy variable: i.e., a one-hot vector of length K,
where all elements are equal to 0.0, except the x-th element which is equal to 1.0
"""
function dummify(x, K)
    oh = zeros(Float64, K)
    oh[x] = 1.0
    return oh
end

"""Novelty function adapted from Xu, Modirshanechi et al. (2020)."""
function novelty(c, K)
    t = collect(1:length(c))
    numer = cumsum(c) .+ 1
    denom = t .+ K
    p = numer ./ denom
    return -log.(p)
end

"""Proportion correct function adapted from Ten et al., (2021)."""
function pc(x, a)
    
end

# Read data
df = @chain CSV.read(DATAPATH, DataFrame) begin
    subset(:condition => ByRow(==("free")))
    subset(:stage => ByRow(==("epochs")))
    select([:pid, :trialsComplete, :famInd, :correct])
end

# Compute features
df = @chain df begin
    transform(:famInd => encode => :famCode)
    # transform(:correct => (x -> Float64.(x)) => :correct)
    groupby(:pid)
    transform(:famCode => (ByRow(x -> dummify(x, K))) => [:c1, :c2, :c3])
    transform([:c1, :c2, :c3] .=> (x -> novelty(x, 3)) .=> [:nov1, :nov2, :nov3])
    groupby(:famCode)
    transform(:correct => (x -> pc(x, a)))
end