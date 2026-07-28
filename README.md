# LAPM: Local Association Prediction Method

This repository contains the reproducible implementation and experiment
settings for **LAPM**, a soil-property prediction method that represents local
spatial association over multiple influence ranges and integrates the resulting
features with coordinates and available covariates through deep regression.

The repository provides:

- generation of the 20 independent random fields used for validation;
- LAPM training, prediction, and MC dropout uncertainty estimation;
- LAPM validation on the generated random fields;
- machine-readable configuration files; and
- source and access information for the two public measured-data cases.

Third-party case data are **not redistributed**. See
[`data/README.md`](data/README.md) for the official sources and access
conditions.

## Method summary

For training coordinates $\mathbf{s}_i$, $K$ spatial centers
$\mathbf{c}_k$ are identified by K-means clustering. The reference bandwidth
$b_0$ is the median nearest-center distance. For bandwidth $b_j$, the
center-specific Gaussian response is

```math
\phi_{k,j}(\mathbf{s}_i)
=
\exp\left[
-\frac{1}{2}
\left(
\frac{\lVert\mathbf{s}_i-\mathbf{c}_k\rVert_2}{b_j}
\right)^2
\right].
```

The $K \times J$ responses are vectorized without pooling and combined with
the coordinates and available covariates:

```math
\mathbf{x}_i
=
\left[
\mathbf{s}_i,\,
\mathbf{v}_i,\,
\operatorname{vec}\{\phi_{k,j}(\mathbf{s}_i)\}
\right].
```

A residual fully connected network learns the nonlinear prediction function.
MC dropout supplies repeated stochastic predictions. The predictive variance
combines variation among stochastic means with the mean conditional variance
returned by the network.

The reported default uses $K=6$ centers and $J=3$ bandwidths:
$b_0/\sqrt{5}$, $b_0$, and $\sqrt{5}b_0$.

## Repository structure

```text
.
├── configs/
│   ├── lapm.json
│   └── random_fields.json
├── data/
│   ├── README.md
│   └── example_schema.csv
├── lapm/
│   ├── io.py
│   ├── metrics.py
│   ├── model.py
│   └── random_fields.py
├── workflows/
│   ├── download_swiss_jura.py
│   ├── generate_random_fields.py
│   ├── run_lapm.py
│   └── run_random_field_lapm.py
├── pyproject.toml
└── requirements.txt
```

The `lapm` directory contains the reusable implementation of the proposed
method. The `workflows` directory contains programs that generate data or run
LAPM. Generated files are written to `data/generated/` or `results/` only when
a workflow is executed; these folders are not stored in the repository.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch automatically uses CUDA when a compatible installation and GPU are
available. The code also runs on CPU.

## Reproduce the random-field experiment

Generate the 20 fields and their leakage-controlled interior splits:

```bash
python workflows/generate_random_fields.py
```

The public field numbers 1-20 reproduce the frozen confirmatory source field
identifiers 65-84. The generator creates independent spectral Matérn-like
covariate and coefficient fields. It does not use the clustering centers or RBF
features fitted by LAPM.

Evaluate LAPM on all 20 generated fields:

```bash
python workflows/run_random_field_lapm.py
```

Run LAPM on a smaller number of fields during development:

```bash
python workflows/run_random_field_lapm.py --fields 2
```

The output contains the MAE, RMSE, coefficient of determination, and mean
predictive standard deviation for each field, together with a summary across
the evaluated fields. Spatial centers, preprocessing transformations, and
network parameters are fitted using the training locations only.

## Run LAPM on local case data

Convert an authorized local copy of a case data set to CSV. The default schema
uses `x`, `y`, one target column, and one or more covariate columns. An empty
header template is provided in `data/example_schema.csv`.

```bash
python workflows/run_lapm.py \
  --train data/external/train.csv \
  --predict data/external/prediction_locations.csv \
  --target target \
  --covariates covariate_1 covariate_2 \
  --output results/lapm_predictions.csv
```

The prediction file must contain coordinates and covariates but does not need a
target column. The output contains `prediction` and `prediction_sd`.

## Core configuration

The manuscript configuration is stored in `configs/lapm.json`. LAPM uses:

| Setting | Value |
|---|---:|
| Spatial centers $K$ | 6 |
| RBF scales $J$ | 3 |
| Hidden feature dimension | 64 |
| Residual blocks | 2 |
| Dropout | 0.15 |
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Training epochs | 180 |
| MC dropout passes | 50 |

The number and spacing of RBF scales can be changed for a new data set, but
selection should be performed inside the training data using spatial
validation.

## Measured-data sources

### German iSOIL data

The German case uses the Bad Lauchstädt gamma-spectrometry data from the iSOIL
project made available through ESDAC. ESDAC requires users to submit the request
form on its official data page:

https://esdac.jrc.ec.europa.eu/content/isoil-project-interactions-between-soil-related-sciences

Because access is request-based, these files are not included and no automated
download script is provided.

### Swiss Jura data

The Swiss Jura data are distributed by the `gstat` project:

https://r-spatial.github.io/gstat/reference/jura.html

Download a pinned and hash-verified copy:

```bash
python workflows/download_swiss_jura.py
```

The downloaded `.rda` file is stored under `data/external/` and remains ignored
by Git.

Full references and field descriptions are provided in
[`data/README.md`](data/README.md).

## Reproducibility notes

- Seeds are fixed in the configuration and workflows.
- K-means centers and all preprocessing transformations are fitted using the
  training subset only.
- Random-field splits contain a spatial buffer between the training and
  interior test locations.
- Results are written to `results/` when a workflow is run. This generated
  folder is ignored by Git.
- The configuration files record the reported manuscript settings; changing a
  setting creates a new experiment and should be documented separately.

## Data and code licensing

The repository does not grant rights to the German iSOIL or Swiss Jura data.
Users must follow the terms of the original providers and cite the original
sources. The authors should add their selected software license before public
release; no license for the source code is assumed merely because the
repository is publicly visible.
