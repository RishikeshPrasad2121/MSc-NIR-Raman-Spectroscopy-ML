# Reproducibility

## Software

MATLAB R2026a

Statistics and Machine Learning Toolbox

## NIR

- 310 samples
- 404 spectral variables
- Approximate range: 7400–10507 cm^-1

## Raman

- 120 samples
- 3401 spectral variables
- Range: 200–3600 cm^-1
- Spectral interval: 1 cm^-1

## Preprocessing

- Raw
- MSC
- SNV
- Savitzky-Golay first derivative
- Savitzky-Golay second derivative

## Savitzky-Golay settings

- Polynomial order: 2
- Window length: 11 spectral points

## Sample selection

The original Kennard-Stone training and test indices retained from the
final MATLAB workspace should be used when available.

## Regression

- Partial Least Squares Regression
- Support Vector Regression
- RBF/Gaussian kernel

## Evaluation

- R²
- RMSEP
- Predicted-versus-reference plots
- Residual analysis
