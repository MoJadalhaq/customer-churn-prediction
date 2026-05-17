import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os


# Maps common churn text labels → 1 (churned) or 0 (retained).
_CHURN_LABEL_MAP = {
    "attrited customer": 1, "churned": 1, "churn": 1, "yes": 1, "true": 1, "1": 1,
    "existing customer": 0, "retained": 0, "no churn": 0, "no": 0, "false": 0, "0": 0,
}


def preprocess_data(df: pd.DataFrame, target_col: str, encoders: dict = None, fit: bool = True):
    """
    Preprocess a DataFrame for churn modeling.

    Returns
    -------
    X          : pd.DataFrame  (encoded feature matrix)
    y          : np.ndarray    (integer labels, 1 = churn)
    encoders   : dict          (LabelEncoder per categorical column)
    """
    df = df.copy().dropna(subset=[target_col]).reset_index(drop=True)

    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    # --- encode target -------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(y_raw):
        lowered = y_raw.astype(str).str.strip().str.lower()
        if lowered.isin(_CHURN_LABEL_MAP).all():
            y = lowered.map(_CHURN_LABEL_MAP).astype(int).values
        else:
            le_y = LabelEncoder()
            y = le_y.fit_transform(y_raw.astype(str)).astype(int)
    else:
        y = y_raw.astype(int).values

    # --- separate column types -----------------------------------------------
    # Use is_numeric_dtype so object, bool, category, and StringDtype are all caught.
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    # --- fill missing values -------------------------------------------------
    for col in num_cols:
        med = X[col].median()
        X[col] = X[col].fillna(med)

    for col in cat_cols:
        X[col] = X[col].fillna("Unknown").astype(str)

    # --- encode categoricals -------------------------------------------------
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    else:
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                # handle unseen labels by mapping them to the most frequent class
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])

    return X, y, encoders


def train_model(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    n_estimators: int = 200,
):
    """
    Train a Random Forest classifier and return everything the dashboard needs.

    Returns
    -------
    model         : fitted RandomForestClassifier
    encoders      : dict of LabelEncoders
    X             : pd.DataFrame  (full preprocessed feature matrix)
    y             : np.ndarray    (full label array)
    report        : dict          (classification_report on held-out test set)
    feature_names : list[str]
    y_test        : np.ndarray    (held-out true labels)
    y_pred_test   : np.ndarray    (held-out predictions)
    y_prob_test   : np.ndarray    (held-out churn probabilities)
    """
    X, y, encoders = preprocess_data(df, target_col)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred_test, output_dict=True)

    return model, encoders, X, y, report, feature_names, y_test, y_pred_test, y_prob_test


def save_model(model, encoders, path: str = "model/churn_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders}, path)


def load_model(path: str = "model/churn_model.joblib"):
    data = joblib.load(path)
    return data["model"], data["encoders"]
