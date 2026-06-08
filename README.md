# Customer Churn Prediction

This project predicts whether a customer is likely to churn (cancel their subscription) using a machine learning model.

The main goal of this project is to build a clean and structured ML project using professional tools from scikit-learn to identify at-risk customers.

---

## Project Overview

This is a binary classification project based on customer subscription data.

The target variable is:

- `Churn`
  - `0` = No Churn (customer stays)
  - `1` = Churn (customer cancels subscription)

The project includes:

- data loading
- data cleaning
- feature selection
- preprocessing with sklearn Pipeline
- handling missing values
- encoding categorical features
- model training
- model evaluation
- model saving with joblib
- prediction on new customer data

---

## Project Structure

```text
customer_churn_project/
│
├── data/
│   └── churn.csv
│
├── models/
│   └── model.joblib
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Features Used

The model uses the following 19 features:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

### Feature Types

Numerical features:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Categorical features:

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

---

## Model

The current model is:

```text
Logistic Regression
```

The model is wrapped inside an sklearn Pipeline together with preprocessing steps.

---

## Preprocessing

The project uses sklearn preprocessing tools.

### Numerical Features

For numerical columns:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Missing values are filled using:

```python
SimpleImputer(strategy="median")
```

Median is used because it is more robust to outliers than mean.

After missing value imputation, numerical features are scaled using:

```python
StandardScaler()
```

The numerical preprocessing pipeline:

```python
SimpleImputer(strategy="median")
↓
StandardScaler()
```

StandardScaler helps put numerical features on a similar scale, which is useful for models such as Logistic Regression.

### Categorical Features

For categorical columns:

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

Missing values are filled using:

```python
SimpleImputer(strategy="most_frequent")
```

Categorical values are encoded using:

```python
OneHotEncoder(drop="first", handle_unknown="ignore")
```

### Data Cleaning

TotalCharges is stored as an object column, so it is converted to numeric:

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
```

Invalid values are converted to missing values and handled by the pipeline.

---

## Pipeline

The project uses an sklearn Pipeline.

The pipeline contains:

```text
preprocessing + model
```

The preprocessing step contains a ColumnTransformer.

The full workflow looks like this:

```text
Raw data
   ↓
ColumnTransformer
   ├── numerical preprocessing
   └── categorical preprocessing
   ↓
Logistic Regression
   ↓
Prediction
```

Using Pipeline helps keep training and prediction consistent.

---

## Why ColumnTransformer?

The customer churn dataset has different types of columns.

Numerical columns and categorical columns need different preprocessing.

For example:

```text
SeniorCitizen, tenure, MonthlyCharges, TotalCharges → numerical preprocessing
gender, Partner, Dependents, etc. → categorical preprocessing
```

ColumnTransformer allows applying different preprocessing steps to different columns inside one object.

---

## Training

To train the model, run:

```bash
python src/train.py
```

The training script does the following:

1. Loads the dataset from `data/churn.csv`
2. Cleans the TotalCharges column
3. Selects features and target
4. Splits data into train and test sets (80/20 split with stratification)
5. Builds the sklearn Pipeline
6. Trains Logistic Regression
7. Evaluates the model
8. Saves the trained Pipeline to `models/model.joblib`

---

## Model Evaluation

The model is evaluated using:

- accuracy
- precision
- recall
- f1-score
- support
- ROC-AUC

Current results:

```text
Accuracy: 0.8055
ROC-AUC: 0.8420
```

Classification report:

```text
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1035
           1       0.66      0.56      0.60       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

---

## Evaluation Notes

The model performs better on class `0` than class `1`.

Class `0` means:

```text
No Churn (customer stays)
```

Class `1` means:

```text
Churn (customer cancels)
```

Recall for class `1` is lower (0.56), which means the model misses some customers who actually churn.

This may happen because the dataset is imbalanced and there are fewer examples of churned customers (374) compared to retained customers (1035).

---

## Prediction

After training, the model is saved as:

```text
models/model.joblib
```

To make a prediction, run:

```bash
python src/predict.py
```

The prediction script does the following:

1. Loads the saved Pipeline from `models/model.joblib`
2. Creates example customer data
3. Makes a prediction
4. Shows churn probability

Example customer:

```text
SeniorCitizen: 0
tenure: 12 months
MonthlyCharges: 70.0
TotalCharges: 840.0
gender: Female
Partner: Yes
Dependents: No
PhoneService: Yes
MultipleLines: No
InternetService: Fiber optic
OnlineSecurity: No
OnlineBackup: Yes
DeviceProtection: No
TechSupport: No
StreamingTV: Yes
StreamingMovies: Yes
Contract: Month-to-month
PaperlessBilling: Yes
PaymentMethod: Electronic check
```

Example output:

```text
Customer Data:
   SeniorCitizen tenure MonthlyCharges TotalCharges gender Partner ...
0             0     12            70.0        840.0 Female    Yes

Predicted class: 1

Churn probability: 72.45%

Result: Customer is likely to churn
```

---

## Predict vs Predict Proba

The project uses both:

```python
model.predict()
```

and:

```python
model.predict_proba()
```

### `predict()`

Returns the predicted class:

```text
0 or 1
```

### `predict_proba()`

Returns probabilities for both classes:

```text
[probability_no_churn, probability_churn]
```

Example:

```text
[0.27, 0.73]
```

This means:

```text
27% probability of no churn
73% probability of churn
```

---

## Saved Model

The trained model is saved using joblib.

```python
joblib.dump(model, "models/model.joblib")
```

Important:

The saved file contains the full Pipeline, not only Logistic Regression.

It includes:

- SimpleImputer (for both numerical and categorical)
- OneHotEncoder
- StandardScaler
- ColumnTransformer
- LogisticRegression

This means `predict.py` does not need to repeat preprocessing manually.

---

## How to Install

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
pandas
scikit-learn
joblib
```

---

## How to Run

### 1. Train the model

```bash
python src/train.py
```

### 2. Make a prediction

```bash
python src/predict.py
```

---

## What I Learned

- How to build a second ML project independently
- How to clean numeric columns stored as text
- How to use Pipeline and ColumnTransformer
- How to evaluate churn classification with ROC-AUC
- Why recall for class 1 is important in churn prediction
- How to save and load a trained model with joblib

---

## Current Status

Completed:

- Professional project structure
- Customer churn baseline model
- sklearn Pipeline
- ColumnTransformer with mixed feature types
- Data cleaning (TotalCharges conversion)
- Missing value handling
- Categorical encoding
- Numerical scaling
- Model evaluation
- Model saving and loading
- Prediction script with example data
- Comprehensive README

---

## Next Steps

- Try class_weight="balanced"
- Tune decision threshold
- Add confusion matrix visualization
- Try RandomForest
- Compare multiple models
- Add cross-validation

---

## Author

Created as part of my Machine Learning learning journey.

Goal:

```text
Build strong ML fundamentals and prepare for ML internship opportunities.
```
