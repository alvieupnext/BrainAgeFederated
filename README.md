# BrainAgeFederated

Federated and centralized training workflows for **brain age estimation** from preprocessed T1 MRI scans using a 3D MONAI DenseNet model.

This project compares:
- centralized training (`centralized.py`)
- federated learning with Flower (`simulation.py`) using `FedAvg` and `FedProx`
- different client data partitioning strategies (dataset-based and synthetic age-distribution-based)

## Thesis Context

This repository is linked to the VUB Master thesis:
- [Federated Learning in Neuroimaging: Pioneering Predictive Models for Brain Age Estimation](https://researchportal.vub.be/en/studentTheses/federated-learning-in-neuroimaging-pioneering-predictive-models-f/)

People listed on the thesis page:
- **Author**: Alvaro Javier Vargas Guerrero
- **Co-authors / promotors**: Guy Nagels, Ann Nowe
- **Supervisor**: Stijn Denissen

## What This Program Is Supposed To Do

At a high level, the code trains a model that predicts chronological age from MRI volumes, then compares centralized and federated performance under different data heterogeneity settings.

Main capabilities:
- Load MRI paths and ages from CSV files
- Train 3D DenseNet models on age regression (L1 loss / MAE-focused)
- Simulate multi-client federated training with Flower
- Save per-round global checkpoints and per-client training logs
- Evaluate trained models and export prediction CSVs
- Plot age distributions, learning curves, and prediction scatter plots

## Repository Contents

### `src/` (Source Code)
- `simulation.py`: Main federated simulation entry point (CLI).
- `client.py`: Flower client implementations (`FlowerClient`, `FedProxClient`) with k-fold local training.
- `strategy.py`: Custom Flower strategies that persist aggregated global checkpoints each round.
- `centralized.py`: Centralized training/testing helpers and evaluation routines.
- `distributions.py`: Synthetic age distribution definitions and dataset sampling logic.
- `plot.py`: Plotting utilities for losses, age distributions, and prediction quality.
- `utils.py`: Shared paths, dataset/model loaders, splitting helpers, model construction.

### `data/` (Datasets)
- `patients_dataset_*.csv`: Metadata CSVs for training/testing splits.

## Data

CSV schema expected by loaders:
- Required columns: `ID`, `processed_file_name`, `Age`, `dataset_name`, `dataset`
- Optional column (present in the 6326 set): `Sex`

Included CSV sizes:
- `patients_dataset_9573.csv`: 9,573 rows
- `patients_dataset_9573_train.csv`: 7,658 rows
- `patients_dataset_9573_test.csv`: 1,915 rows
- `patients_dataset_6326.csv`: 6,326 rows
- `patients_dataset_6326_train.csv`: 5,060 rows
- `patients_dataset_6326_test.csv`: 1,266 rows

### Client Data Splitting Strategies
When running federated simulations, you must specify how to divide the data amongst the simulated clients (`--split`):
- **Dataset Split (`--split dataset`)**: This partitions the data based on the actual clinical origin of the MRI scans (using the `dataset_name` column). This mimics a real-world cross-silo scenario where each client represents a specific hospital or clinical trial dataset.
- **Distribution Split (`--split distribution`)**: This artificially shapes the dataset each client receives based on synthetic statistical age distributions (e.g., Gaussian mixtures). This allows you to aggressively test how the algorithm handles extreme age imbalances (where some clients only see younger brains and others see elderly brains) regardless of the hospital source.

Important:
- `processed_file_name` values are absolute MRI file paths. You need those NIfTI files available in your environment.
- Default training/testing file names used by code are configured in `utils.py`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `torch`, `monai`, `flwr`, `ray`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `nibabel`.

## Running Federated Simulation

Default run (FedAvg, dataset split):

```bash
python src/simulation.py \
  --strategy FedAvg \
  --split dataset \
  --server_rounds 5 \
  --kcrossval 10 \
  --nodes 3
```

Example with FedProx + DWood initialization + distribution split:

```bash
python src/simulation.py \
  --strategy FedProx \
  --seed 2 \
  --split distribution \
  --distribution Gaussian \
  --server_rounds 5 \
  --kcrossval 10 \
  --nodes 3 \
  --alias 3_Node
```

Notes:
- `--seed <n>` switches initialization from random weights (`RW`) to DWood checkpoint mode and expects `./utils/models/DWood/T1/seed_<n>.pt`.
- `--distribution Transition` is defined for 6-node profiles.

## Running Centralized Baseline

`src/centralized.py` is function-driven (no full CLI parser). Example calls:

```bash
python -c "import sys; sys.path.append('src'); from centralized import run_model; run_model('centralized_RW', epochs=20, kcrossval=10, test_dataset='data/patients_dataset_9573_test.csv')"
```

```bash
python -c "import sys; sys.path.append('src'); from centralized import run_model; run_model('centralized_DWood', epochs=20, kcrossval=10, seed='./utils/models/DWood/T1/seed_2.pt', test_dataset='data/patients_dataset_9573_test.csv')"
```

## Outputs

Typical artifacts are written under:
- `./utils/models/<project_name>/`
- `./utils/tests/<project_name>/`
- `./utils/plots/`

Examples:
- `federated_model_round_<r>.pt` (global federated checkpoints)
- `<client>/<client>_losses.txt` or CSV fold logs
- `<client>/patient_ids.txt`
- `centralized_losses.txt` / `centralized_mae.txt`
- `<project_name>_brain_age_output.csv` predictions
- PDF plots for distributions and training/evaluation curves

## Citation

If you use this code, cite the thesis above and acknowledge the listed author/co-authors/supervisor.
