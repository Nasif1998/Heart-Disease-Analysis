# Heart-Disease-Analysis - Machine Learning vs Deep Learning

## Overview

This project investigates heart disease prediction using structured clinical data.
The goal is to compare classical machine learning models with a neural network (MLP) and evaluate whether increased model complexity improves performance on tabular medical data.

The comparison is performed using a leakage-safe pipeline with proper train/validation/test separation.

## Dataset

Dataset: heart.csv
Task: Binary Classification
0 → No disease
1 → Disease

Features include:
Age
Cholesterol
Blood pressure
Chest pain type
ECG results
And other clinical indicators

## Methodology

### Data Splitting
- Train: 64%
- Validation: 16%
- Test: 20%
- Stratified splitting used to preserve class distribution.

### Preprocessing (Leakage-Safe)
Implemented using Pipeline and ColumnTransformer.
- Numerical features:
     - Median imputation
     - Standard scaling
- Categorical features:
     - Most frequent imputation
     - One-hot encoding (handle_unknown='ignore')

All preprocessing is fitted only on training data.

## Classical Machine Learning Models

### Logistic Regression
 - Tuned hyperparameter: C
 - Class imbalance handled via class_weight='balanced'

### Random Forest

- Tuned:
    - n_estimators
    - max_depth

### XGBoost

- Tuned:
    - max_depth
    - learning_rate
    - subsample
    - colsample_bytree

Model selection based on highest validation ROC-AUC using 5-fold stratified cross-validation.

## Deep Learning Model

Two MLP architectures implemented in PyTorch:

### MLP_small
   - Hidden layers: [64, 32]

### MLP_BN
   - Hidden layers: [128, 64]
   - Batch Normalization
   - Dropout (0.3)
   - Training setup:
       - Optimizer: Adam
       - Loss: BCEWithLogitsLoss
       - Batch size: 32
       - Epochs: 150
       - Learning rate sweep: {1e-4, 1e-3, 1e-2}

Best model selected using validation ROC-AUC.

## Threshold Selection

Instead of using a fixed 0.5 threshold:
- Thresholds from 0.05 to 0.95 evaluated on validation set
- F1-score maximized
- Selected threshold applied to test set

## Evaluation Metrics

Final evaluation performed once on the test set:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC & PR Curves

## Results

| Model                      | Test ROC-AUC | Test F1 | Notes                              |
| -------------------------- | ------------ | ------- | ----------------------------------- |
| Logistic Regression (Best ML) | **0.943**    | **0.733** | Strong linear baseline              |
| MLP_BN (Best DL)          | 0.865        | 0.444   | Regularized neural network          |

## Conclusion

This project demonstrates that for structured clinical tabular data, well-regularized classical machine learning models can match or outperform shallow neural networks when evaluated using proper validation protocols.


