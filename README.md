# Explainable Active Learning with Intervention

This repository contains the implementation of the explainable active learning framework proposed in the paper 'Why Does This Query Need to Be Labeled?: Enhancing Active Learning through Explanation-Based Interventions in Query Selection'

## Overview

Our framework decomposes the acquisition function into feature-level contributions using SHAP, and allows labelers to adjust these contributions via a feature-weighted intervention.

## Dependencies

* numpy
* pandas
* scikit-learn
* shap
* scipy


## File Descriptions

* **run\_experiment.py**: Entry point script. Parses command-line arguments (`--dataname`, `--method`, `--lambda_val`, `--rep`) and invokes the active learning loop. Saves results to a pickle file.
* **data\_utils.py**: Utility functions to load datasets (`dataset_load_all`) and inject feature noise (`noise_injection_uni_simple`).
* **model\_utils.py**: Functions for training the prediction model (`train_model`), reordering prediction probabilities to match the full class set (`predict_proba_ordered`), and computing evaluation metrics (`my_scores_with_scaler`).
* **explainer\_utils.py**: Implements margin-based uncertainty scoring (`UncertaintyScorer`) and SHAP-based adjusted acquisition scoring (`get_acquisition_scores`) for feature-weighted intervention.
* **al\_loop.py**: Contains `run_active_learning`, which orchestrates the active learning process: initial train/test split, iterative query selection, model updates, and metric collection.

**Note**: Place dataset CSV files under a datasets/ directory in the project root, with each file named \<dataname>.csv.

## Usage

### Single experiment

Run a single AL experiment on one dataset, method, and repetition index:

```bash
python run_experiment.py --dataname avila --method al_ei --lambda_val 0.3 --rep 0
```

* `--dataname`: name of the CSV file (without `.csv`) in `datasets/`.
* `--method`: `al` (conventional) or `al_ei` (our method).
* `--lambda_val`: weight for noisy features when `method=al_ei` (default `0.0`).
* `--rep`: random seed / repetition index.

Results will be saved as `results_<dataname>_<method>_<lambda_val>_<rep>.pkl`.

### Batch runs

You can script over multiple datasets/methods by writing a simple Python or shell loop:

```bash
# Example: run both methods on multiple datasets
for d in avila magic vowel; do
  python run_experiment.py --dataname $d --method al --rep 0
  python run_experiment.py --dataname $d --method al_ei --lambda_val 0.0 --rep 0
done
```
