import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    r2_score,
    mean_squared_error
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import shap


# -------------------------------------------------
# 1️⃣ SAFE CHURN PREDICTION (LOGISTIC REGRESSION)
# -------------------------------------------------
def churn_prediction(df):

    df = df.copy()

    if len(df) < 5:
        return {"error": "Dataset too small for churn modeling."}

    threshold = df['Recency'].median()
    df['Churn'] = df['Recency'] > threshold

    # Check class distribution
    class_counts = df['Churn'].value_counts()

    if len(class_counts) < 2 or class_counts.min() < 2:
        return {"error": "Not enough samples in one class for modeling."}

    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Churn']

    # Avoid stratify for small datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    # Safe cross-validation
    cv_auc = None
    if len(df) >= 10:
        cv_auc = cross_val_score(
            pipeline, X, y, cv=3, scoring='roc_auc'
        ).mean()

    return {
        "model": pipeline,
        "classification_report": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "roc_auc": auc,
        "cross_val_auc": cv_auc,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_probs": probs
    }


# -------------------------------------------------
# 2️⃣ SAFE RANDOM FOREST MODEL
# -------------------------------------------------
def random_forest_churn(df):

    df = df.copy()

    if len(df) < 5:
        return {"error": "Dataset too small for churn modeling."}

    threshold = df['Recency'].median()
    df['Churn'] = df['Recency'] > threshold

    class_counts = df['Churn'].value_counts()

    if len(class_counts) < 2 or class_counts.min() < 2:
        return {"error": "Not enough samples in one class for modeling."}

    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    feature_importance = dict(
        zip(X.columns, model.feature_importances_)
    )

    return {
        "model": model,
        "roc_auc": auc,
        "feature_importance": feature_importance,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_probs": probs
    }


# -------------------------------------------------
# 3️⃣ SAFE SALES REGRESSION MODEL
# -------------------------------------------------
def sales_regression_model(df):

    df = df.copy()

    if len(df) < 5:
        return {"error": "Dataset too small for regression modeling."}

    df['Month'] = df['OrderDate'].dt.month

    X = df[['Month', 'Quantity']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    cv_r2 = None
    if len(df) >= 10:
        cv_r2 = cross_val_score(
            model, X, y, cv=3, scoring='r2'
        ).mean()

    return {
        "model": model,
        "r2_score": r2,
        "rmse": rmse,
        "cross_val_r2": cv_r2
    }


# -------------------------------------------------
# 4️⃣ ROC CURVE DATA
# -------------------------------------------------
def roc_curve_data(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    return fpr, tpr, thresholds


# -------------------------------------------------
# 5️⃣ SHAP EXPLAINABILITY
# -------------------------------------------------
def shap_analysis(model, X):

    if model is None:
        print("SHAP skipped: No model available.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[1], X)
    except Exception as e:
        print("SHAP analysis skipped:", e)
