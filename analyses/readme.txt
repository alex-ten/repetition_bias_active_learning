1. Model fitting
    run script `julia --threads=4 analyses/fit_choice_models.jl`
2. Diagnostics
    2.1. `data_prep/diagnostics.jl` creates the diagnostics csv using in Figure 1. 
    2.2. The script `visualization/exclusions.jl` plots the data.
3. Comparisons
    3.1. `prep_data/comparisons.jl` creates the compariosns csv (excluding data from problematic chains)
    3.2. `analyses/comparisons.jl` performs the linear regression of predictors on accuracy.
        3.2.1. There is also a script for performing the regression with the number of switches.
    3.3. `visualizations/elpd_diffs.jl` generates figure 3 (model comparisons). It requires a comparisons datataset from 3.1.
    3.4. `visualizations/elpd_byNumSwitch.jl` generates figure 4. It requires a comparisons datataset from 3.1.
