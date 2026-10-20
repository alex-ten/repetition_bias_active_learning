# The role of repetition bias in self-directed learning

Data and Julia code for the analyses, model fits and figures. All scripts assume they are run from the repository root and activate the `Blocking` project environment (`using Pkg; Pkg.activate("Blocking")`).

**Requires Julia 1.11.** The environment was resolved with Julia 1.11.8; newer versions fail to load some packages. With juliaup: `juliaup add 1.11`, then run `juliaup override set 1.11` in the repository root (or start Julia with `julia +1.11`).

## Authorship

All code was written by Alex Ten. The repository was tidied up for publication with the help of Claude (Anthropic).

## Repository structure

| Folder / file | Contents |
|---|---|
| `data/` | Behavioral data and derived datasets. `raw_data/` holds the raw experiment exports per condition (`blocked`, `free`, `interleaved`, `post`) and the merged `combined*.csv`. Top-level CSVs are derived: choice-model features (`df_features.csv`), convergence diagnostics, model comparisons, posterior summaries. |
| `data_prep/` | Scripts that turn raw data or model fits into the derived datasets in `data/` (features, diagnostics, comparisons, behavior summaries). |
| `cluster/` | Model-fitting code used on the HPC cluster: settings and `fit_one` (`fit_helpers.jl`), the generated model definitions (`models.jl`, from `generate_models.jl`) and the job table of 121 participants × 31 models (`lookup.csv`, from `jobarrayconfig.jl`). |
| `analyses/` | Statistical analyses (regressions, learning, confidence, survival model, descriptives) and local versions of the fitting routines for testing. `analyses/readme.txt` lists the steps of the main pipeline. |
| `simulations/` | Simulations of choice behavior and their outputs. |
| `visualization/` | Scripts for the figures, plus the summary CSVs some of them read. |
| `plots/` | Generated figures and tables. |
| `Blocking/` | Julia project environment (`Project.toml`, `Manifest.toml`) and a small helper module. |
| `lib.jl`, `vislib.jl` | Shared helper functions for feature computation/simulation and for plotting/reporting. |

## Caveats

- **Cluster code is incomplete.** The job-array runner script that called `fit_one` on the cluster, and its logs, are not in the repository yet. `cluster/` holds the configuration, model definitions and fitting function, but not the script that dispatches jobs.
- **MCMC fits are not included.** The per-participant posterior samples (`data/comparisons/`, `data/comparisons_6000_frozen/`, about 300 GB of NetCDF files) are too large to share. The derived CSVs in `data/` hold everything the analyses and figures need. Scripts in `data_prep/` that read the `.nc` files cannot be rerun without them.
- **Participants are anonymized.** Participant IDs are integers (`pid` 1–370). Participants 1–121 are the free-choice sample used for model fitting. Prolific IDs are not included; the demographic exports (`data/raw_data/*/*_demo.csv`) keep only demographics and are stripped of submission IDs, completion codes and timestamps.
