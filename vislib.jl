
function spatial_entropy(x::AbstractArray{Int})
    x = ndims(x) == 1 ? reshape(x, 1, :) : x
    
    hs = Float64[]
    
    for row in eachrow(x)
        classes = unique(row)
        k = length(classes)
        coords = Float64.(1:length(row))
        
        ps = Float64[]
        edges = Int[]
        centroids = Float64[]
        
        for c in classes
            instances = row .== c
            
            # Calculate proportions
            push!(ps, mean(instances))
            
            # Count edges
            push!(edges, sum(instances[1:end-1] .!= instances[2:end]))
            
            # Calculate centroids
            coords_copy = copy(coords)
            coords_copy[.!instances] .= NaN
            push!(centroids, mean(filter(!isnan, coords_copy)))
        end
        
        dists = Float64[]
        for i in 1:k
            push!(dists, sum(abs.(centroids[i] .- centroids[setdiff(1:k, i)])))
        end
        
        push!(hs, -sum((edges .* dists) .* ps .* log.(ps)))
    end
    
    return hs
end

function mark_switches(x::AbstractArray{Int})
    switch = zero(x)
    switch[2:end] = x[1:end-1] .!= x[2:end]
    return switch
end

function addcols(gdf)
    return @chain gdf begin
        DataFrames.transform(:famCode => mark_switches => :switch; ungroup=false)
        DataFrames.transform(:switch => cumsum => :blockNum)
        groupby([:pid, :blockNum])
        DataFrames.transform(:blockNum => length => :blockSize)
    end
end

function summarize(df)
    return @chain df begin    
        combine(
            :famCode => spatial_entropy => "spEnt",
            :switch => sum => :numSwitches,
            :blockSize => var => :blockSizeVar
        )
    end
end