import pandas as pd
import numpy as np
import joblib 


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_validate

DATA_PATH = 'data/churn.csv'
MODEL_PATH = "models/model.joblib"


NUM_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",

    # Engineered features
    "AverageMonthlySpend",
    "IsNewCustomer",
    "HasLongTermContract",
    "HasSupportServices",
    "IsHighMonthlyCharge",
    "TotalServices",
]

CAT_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUM_FEATURES+CAT_FEATURES

TARGET = 'Churn'

def load_data(data_path):
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def features_preparation(df,features,target):
    X = df[features]
    y = df[target].map({"No":0, "Yes":1})

    return X,y


def add_features(df):
    df = df.copy()
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    df["AverageMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)
    df["HasLongTermContract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    df["HasSupportServices"] = (
        (df["OnlineSecurity"] == "Yes") | 
        (df["TechSupport"] == "Yes")
    ).astype(int)
    
    monthly_median = df["MonthlyCharges"].median()
    df["IsHighMonthlyCharge"] = (df["MonthlyCharges"] > monthly_median).astype(int)
    
    service_cols = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)
    
    return df


def build_pipeline(classifier):
    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer,NUM_FEATURES),
            ("cat", cat_transformer, CAT_FEATURES)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test,y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print()
    print('Classification Report')
    print(classification_report(y_test,y_pred))


def save_model(model,model_path):
    #Save trained model pipeline

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def compare_models(models, X, y):
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "recall": "recall",
        "f1": "f1"
    }

    for name, classifier in models.items():
        print("=" * 50)
        print(name)

        model = build_pipeline(classifier)

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )

        for metric in scoring.keys():
            scores = cv_results[f"test_{metric}"]
            print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")



def main():

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest" : RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "Gradient Boosting" : GradientBoostingClassifier(
            random_state=42
        ) 
    }

    df = load_data(DATA_PATH)

    df = add_features(df)

    X,y = features_preparation(df, FEATURES, TARGET)

    compare_models(models,X,y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42, stratify=y)

    best_model = build_pipeline(models["Logistic Regression"])

    best_model = train_model(best_model,X_train, y_train)

    evaluate_model(best_model, X_test, y_test)

    save_model(best_model, MODEL_PATH)





if __name__ == "__main__":
    main()