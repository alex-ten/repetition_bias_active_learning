1. Model fitting
    Main fits run on the cluster: `cluster/jobarrayconfig.jl` builds `cluster/lookup.csv` (121 pids x 31 models),
    `cluster/generate_models.jl` builds `cluster/models.jl`, and `cluster/fit_helpers.jl` holds the settings and `fit_one`.
    Outputs go to `data/comparisons/pid-<pid>-<pid_str>/m-<mt>-<label>.nc`.
    Local counterparts (for tests) reuse the cluster code; run from the repository root:
    1.1. `julia --threads=4 analyses/fit_choice_models.jl` fits all (or a subset of) pids and models.
    1.2. `analyses/fit_choice_models_manual.jl` refits the models missing for one pid.
    1.3. `analyses/check_fit_data.jl` lists missing fits in `data/fit_errors.csv`.
2. Diagnostics
    2.1. `data_prep/diagnostics.jl` creates the diagnostics csv using in Figure 1. 
    2.2. The script `visualization/exclusions.jl` plots the data.
3. Comparisons
    3.1. `prep_data/comparisons.jl` creates the compariosns csv (excluding data from problematic chains)
    3.2. `analyses/comparisons.jl` performs the linear regression of predictors on accuracy.
        3.2.1. There is also a script for performing the regression with the number of switches.
    3.3. `visualizations/elpd_diffs.jl` generates figure 3 (model comparisons). It requires a comparisons datataset from 3.1.
