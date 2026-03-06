# Profundidad efectiva en algoritmos de Quantum Machine Learning: atenuación del gradiente en VQAs bajo ruido NISQ

## Authors

Carlos Mario Braga\
Manuel A. Serrano\
Eduardo Fernández-Medina

**Contact:** carlosmario.braga1@alu.uclm.es

Universidad de Castilla-La Mancha, Ciudad Real, Spain

------------------------------------------------------------------------

## Overview

This repository contains the full experimental artifact accompanying the
article:

> **Profundidad efectiva en algoritmos de Quantum Machine Learning:
> atenuación del gradiente en VQAs bajo ruido NISQ**

The goal of the repository is to:

-   Provide full reproducibility of the experimental pipeline.
-   Make explicit the structural assumptions behind the effective
    attenuation model.
-   Expose the computational workflow used to estimate the effective
    attenuation rate ( `\epsilon`{=tex}\_{`\text{eff}`{=tex}} ).
-   Extend and clarify the experimental artifact beyond what is
    described in the paper.

The repository implements the complete pipeline:

1.  Experiment execution under NISQ-like noise.
2.  Multi-run consolidation and statistical analysis.
3.  Regime-based structural analysis.
4.  Estimation of the effective gradient attenuation rate.

------------------------------------------------------------------------

# Conceptual Structure of the Artifact

The repository is organized around the central hypothesis:

\|E\[∇\]\|(L) ∝ e\^{-ε_eff L}

and its empirical operationalization using:

V̄\^{(K)}\_∇

where:

-   L = circuit depth (number of layers)
-   K = number of final epochs used to compute the gradient variance
    proxy
-   ε_eff = effective structural attenuation rate per layer

------------------------------------------------------------------------

# Pipeline Overview

1)  qml_cybernetic_experiment.py\
2)  collect_results_from_dirs_with_stats.py\
3)  analyze_experiment_regimes_batch.py\
4)  estimator.py

Each stage consumes the outputs of the previous one.

------------------------------------------------------------------------

# 1. Running the Experiments

Script: qml_cybernetic_experiment.py

Example:

``` bash
nohup python3 qml_cybernetic_experiment.py \
    --n_qubits 6 \
    --n_layers 5 \
    --epochs 60 \
    --shots 60 \
    --lr 0.05 \
    --seeds 0 1 2 3 4 5 \
    --out_prefix est_model_q6_l5_s60_heavy \
    --n_samples 512 \
    --data_noise 0.05 \
    --test_size 0.25 \
    --noise_kind heavy &
```

Generates:

-   Per-seed JSON files (loss, gradient variance, controller actions)
-   summary_results.csv

------------------------------------------------------------------------

# 2. Consolidating Results

Script: collect_results_from_dirs_with_stats.py

Example:

``` bash
python collect_results_from_dirs_with_stats.py \
    --root_dir . \
    --outdir results_summary_round5 \
    --last_k 10
```

Important parameter:

--last_k defines K in V̄\^{(K)}\_∇, the structural gradient variance
proxy used throughout the paper.

Outputs:

-   combined_per_seed.csv
-   summary_aggregated.csv
-   stats_tests.csv

------------------------------------------------------------------------

# 3. Regime-Based Analysis

Script: analyze_experiment_regimes_batch.py

Example:

``` bash
python analyze_experiment_regimes_batch.py \
    --csv results_summary_round5/combined_per_seed.csv \
    --outdir results_analysis_round5_heavy \
    --filter_noise heavy \
    --n_boot 5000
```

Generates per (q, shots) folders with:

-   deltas_per_seed.csv
-   stats_by_L.csv
-   Figures (delta vs L, gradient variance vs L, heatmaps, etc.)

------------------------------------------------------------------------

# 4. Estimating ε_eff

Script: estimator.py

Example:

``` bash
python3 estimator.py \
    --input results_analysis_round5_heavy/
```

Implements OLS:

log(avg_grad_var_lastK) = β0 − ε_eff L

Outputs:

-   epsilon_estimates.csv
-   epsilon_estimates.json

------------------------------------------------------------------------

# Reproducibility Notes

-   All randomness is controlled via fixed seeds.
-   Bootstrap procedures use fixed RNG seeds.
-   The full pipeline is deterministic once JSON outputs are generated.

------------------------------------------------------------------------

# Contact

For questions regarding the experimental artifact or reproducibility:

**Carlos Mario Braga**\
carlosmario.braga1@alu.uclm.es
