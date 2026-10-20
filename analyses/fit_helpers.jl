function get_params_matrix(chain, m)
    df = DataFrame(mean(chain))
    # bvec = DataFrame(mean(chain)).mean
    q = m - 1
    d = 1 + q + m * 1
    bmat = Matrix{Float64}(undef, q, d)

    for (j, jsub) in "₁₂" |> enumerate
        # Intercepts
        α = subset(df, :parameters => ByRow(x -> occursin("α$jsub", x |> String)))[:, :mean]

        # Lagged choice params
        β = subset(df, :parameters => ByRow(x -> occursin("β$jsub", x |> String)))[:, :mean]
        
        # Covariate params
        γ = subset(df, :parameters => ByRow(x -> occursin("γ$jsub", x |> String)))[:, :mean]

        bmat[j, :] .= vcat(α, β, γ)
    end
    return bmat
end

function addlag(df, col, lag)
    res = df[:, :]
    res[1:end-lag, col] .= df[1+lag:end, col]
    return res
end

function zscore(col)
    m = mean(col)
    return (col .- m) ./ std(col; corrected=true, mean=m)
end

function unique_index_combinations(indices)
    unique!([sort([i for (j,i) in enumerate(indices) if (k & (1 << (j-1))) != 0])
             for k in 1:(1<<length(indices))-1])
end

# Given a list of variable names, make a new list with these names repeated n times and indexed 1:n
function rep_vars(vars::Vector{Symbol}, n::Int)
    res = []
    for v in vars
        for i in 1:n
            push!(res, "$(v)$(i)")
        end
    end
    return Symbol.(res)
end

# Extract data for a specific model
function prep_data(df::DataFrame, pid::S, vars::Vector{Symbol}) where S <: AbstractString
    res = @chain df begin
        DataFrames.subset(:pid => ByRow(==(pid)))
        addlag(_, :famCode, 1)
        first(nrow(df) - 1)
        DataFrames.select(:famCode, vars...)
    end
    return res
end

function confidence_interval_95(estimate, standard_error)
    # For a 95% CI, we use approximately 1.96 standard deviations
    # (more precisely, it's the 97.5th percentile of the standard normal distribution)
    z_score = quantile(Normal(), 0.975)
    
    margin_of_error = z_score * standard_error
    
    lower_bound = estimate - margin_of_error
    upper_bound = estimate + margin_of_error
    
    return (lower_bound, upper_bound)
end
