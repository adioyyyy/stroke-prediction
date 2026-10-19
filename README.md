# Stroke Prediction using Logistic Regression
## Author
ABDISALAM NUR

BTEC International Level 3 Diploma in IT: Introduction to Artificial Intelligence (AI)

## Project overview
This project builds an AI model that predicts whether a patient is likely to have a stroke, to support healthcare decision-making. It uses the Kaggle **Stroke Prediction Dataset** (5,110 patients, 11 features) and a **Logistic Regression** model built with Python.

## Objectives
- Predict stroke risk from patient data (age, hypertension, heart disease, glucose level, BMI, smoking status, etc.)
- Prioritise **recall** (catching real stroke cases) while keeping precision and overall performance reasonable
- Compare a baseline model against refined models and evaluate their effectiveness

## Dataset
- Source: Kaggle, "Stroke Prediction Dataset" (`healthcare-dataset-stroke-data.csv`)
- Only about 4.9% of patients had a stroke (249 of 5,110), so the classes are heavily imbalanced

## Method
1. **Data preparation:** drop `id`, remove rare "Other" gender, fill missing BMI (201 values) with the training median, encode categorical features, stratified 80/20 train/test split, scale numeric features
2. **Class imbalance:** oversample the minority class in the training set only
3. **Baseline model:** Logistic Regression on the prepared data
4. **Refinement:** class weights, hyperparameter tuning with GridSearchCV (C, penalty), and decision threshold tuning
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | 0.95 | 1.00 | 0.02 | 0.04 |
| + Oversampling | 0.74 | 0.13 | 0.80 | 0.23 |
| + Class weights and tuning | 0.74 | 0.13 | 0.80 | 0.23 |
| Final (tuned + threshold) | 0.88 | 0.23 | 0.64 | 0.34 |

The baseline had high accuracy but missed almost every stroke. The final model detects 32 of 50 strokes in the test set.

## How to run
1. Open `stroke_prediction_full.ipynb` in Google Colab or Jupyter
2. Upload `healthcare-dataset-stroke-data.csv` when prompted (or place it in the same folder)
3. Run all cells in order

Install requirements locally with:
```
pip install -r requirements.txt
```

## Files
- `stroke_prediction_full.ipynb`: full notebook with code and outputs
- `healthcare-dataset-stroke-data.csv`: dataset
- `requirements.txt`: Python libraries
- `README.md`: project overview

## Author
ABDISALAM NUR, Oshwal College
