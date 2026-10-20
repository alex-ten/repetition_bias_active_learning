module Blocking

using Turing
using FillArrays
using NNlib: softmax

include("model.jl")

export novelty

end
