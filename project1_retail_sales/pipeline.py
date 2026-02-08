import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------
# 1️⃣ NUMERIC PIPELINE
# -------------------------------------------------
def create_numeric_pipeline():

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return numeric_pipeline


# -------------------------------------------------
# 2️⃣ CATEGORICAL PIPELINE
# -------------------------------------------------
def create_categorical_pipeline():

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return categorical_pipeline


# -------------------------------------------------
# 3️⃣ FULL PREPROCESSOR (ColumnTransformer)
# -------------------------------------------------
def create_preprocessor(df):

    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = create_numeric_pipeline()
    categorical_pipeline = create_categorical_pipeline()

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor


# -------------------------------------------------
# 4️⃣ FULL ML PIPELINE (PREPROCESSING + MODEL)
# -------------------------------------------------
def create_full_pipeline(df, model_type="logistic"):

    preprocessor = create_preprocessor(df)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    full_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    return full_pipeline


# -------------------------------------------------
# 5️⃣ OPTIONAL: OUTLIER HANDLING FUNCTION
# -------------------------------------------------
def remove_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_clean
