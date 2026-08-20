# Machine Learning Modelling of Spectroscopic Measurements of Pharmaceuticals

## MSc Pharmaceutical Science – University of Surrey

This repository contains the MATLAB analysis workflow and supporting materials for an MSc research project investigating the prediction of active pharmaceutical ingredient (API) concentration in pharmaceutical tablets using Near-Infrared (NIR) and Fourier-transform Raman spectroscopy.

## Author

**Rishikesh Prasad**

## Supervisor

**Professor Tao Chen**

## Project overview

The study investigated the effect of spectral preprocessing and regression modelling approaches on the prediction of API concentration from pharmaceutical tablet spectra.

Two spectroscopic datasets were analysed:

- **NIR transmittance:** 310 tablet samples and 404 spectral variables
- **FT-Raman spectroscopy:** 120 tablet samples and 3401 spectral variables

The datasets were obtained from the UCPH Chemometrics Research Database and were originally reported by Dyrby et al. (2002).

The original datasets are **not redistributed in this repository**.

## Analytical workflow

The analysis included:

1. Data import and preparation
2. Spectral preprocessing
3. Exploratory spectral analysis
4. Principal Component Analysis (PCA)
5. Q-residual analysis
6. Kennard-Stone training/test-set selection
7. Partial Least Squares Regression (PLSR)
8. Support Vector Regression (SVR)
9. Independent test-set evaluation
10. Residual analysis

## Spectral preprocessing

Five spectral representations were investigated:

- Raw spectra
- Multiplicative Scatter Correction (MSC)
- Standard Normal Variate (SNV)
- Savitzky-Golay first derivative (SG1)
- Savitzky-Golay second derivative (SG2)

For the Savitzky-Golay transformations, a second-order polynomial and an 11-point window were used.

## Regression modelling

Two regression approaches were evaluated:

- Partial Least Squares Regression (PLSR)
- Support Vector Regression (SVR)

The SVR models used a Gaussian/RBF kernel. The final Raman SG1 and SG2 SVR configurations used a KernelScale value of 50, while the other SVR configurations used the documented automatic/default parameter settings.

## Model evaluation

Performance was assessed using:

- R²
- RMSEP
- Predicted-versus-reference plots
- Residual analysis

The same independent Kennard-Stone test samples were used for model comparison within each dataset.

## Software

The analysis was performed using:

**MATLAB R2026a**

with the:

**Statistics and Machine Learning Toolbox**

## Repository structure

```text
DOCUMENTATION/
├── Data_Source.md
├── Reproducibility.md
└── Workflow.md

NIR/
├── CODE/
├── FIGURES/
│   ├── MSC/
│   ├── RAW/
│   ├── SGdata/
│   └── SNV/
└── RESULTS/

RAMAN/
├── CODE/
├── FIGURES/
│   ├── MSC/
│   ├── RAW/
│   ├── SGdata/
│   └── SNV/
└── RESULTS/
