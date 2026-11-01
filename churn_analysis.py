"""
Реальный анализ оттока клиентов на датасете IBM Telco Customer Churn.
- 7043 записи, 21 признак
- XGBoost с stratified 5-fold кросс-валидацией
- Permutation importance для интерпретации
- Бизнес-инсайты: churn rate по сегментам

Источник датасета: https://github.com/IBM/telco-customer-churn-on-icp4d
(публичный датасет IBM, не требует API-ключа Kaggle)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# 1. Загрузка и чистка данных
# ---------------------------------------------------------------------------
RAW_PATH = Path("/home/z/my-project/data/raw/telco-churn.csv")
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} rows × {len(df.columns)} cols")

# Удаляем customerID — бесполезен для ML
df = df.drop(columns=["customerID"])

# TotalCharges идёт как строка с пустыми значениями " "
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(f"Missing TotalCharges: {n_missing}")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
print(f"After cleanup: {len(df)} rows")

# Целевая переменная → 0/1
df["Churn"] = (df["Churn"] == "Yes").astype(int)
churn_rate = df["Churn"].mean()
print(f"Overall churn rate: {churn_rate:.2%}")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
# Бинарные Yes/No → 0/1
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = (df[col] == "Yes").astype(int)

# SeniorCitizen уже 0/1

# Категориальные колонки — оставляем как есть, one-hot будет в pipeline
categorical_cols = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Числовые признаки
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Доп. фича: средний чек за месяц (TotalCharges / tenure)
df["AvgMonthlyCharges"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)
numeric_cols.append("AvgMonthlyCharges")

# Доп. фича: длительность контракта в месяцах
contract_months_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
df["ContractMonths"] = df["Contract"].map(contract_months_map)
numeric_cols.append("ContractMonths")

# Доп. фича: есть ли доп. сервисы (количество подписок)
extra_services = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
df["ExtraServicesCount"] = (df[extra_services] == "Yes").sum(axis=1)
numeric_cols.append("ExtraServicesCount")

print(f"Final feature count: {len(numeric_cols) + len(categorical_cols) + len(binary_cols)}")

# ---------------------------------------------------------------------------
# 3. Подготовка матрицы признаков
# ---------------------------------------------------------------------------
feature_cols = numeric_cols + binary_cols + categorical_cols
X = df[feature_cols].copy()
y = df["Churn"].copy()

# ---------------------------------------------------------------------------
# 4. XGBoost с stratified 5-fold кросс-валидацией
# ---------------------------------------------------------------------------
# One-hot encoding через ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols + binary_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# XGBoost с балансировкой классов
scale_pos_weight = (len(y) - y.sum()) / y.sum()
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb)])

# Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
cv_recall = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1)
cv_precision = cross_val_score(pipeline, X, y, cv=cv, scoring="precision", n_jobs=-1)
cv_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)

print("\n=== XGBoost 5-fold CV ===")
print(f"ROC-AUC:   {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"Accuracy:  {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"F1:        {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ---------------------------------------------------------------------------
# 5. Финальная модель на train/test split для permutation importance
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline.fit(X_train, y_train)

# ROC-AUC на test
from sklearn.metrics import roc_auc_score as _auc
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)
test_auc = _auc(y_test, y_pred_proba)
test_acc = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("\n=== Final model on held-out test set ===")
print(f"ROC-AUC:   {test_auc:.4f}")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"F1:        {test_f1:.4f}")
print("\nConfusion matrix (TN, FP, FN, TP):")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Permutation importance (на test set, real feature importance)
# ---------------------------------------------------------------------------
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc"
)

imp_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

print("\n=== Permutation importance (top 10) ===")
print(imp_df.head(10).to_string(index=False))

# Бизнес-инсайт: churn rate по ключевым сегментам
segment_churn = {}
for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
    seg = df.groupby(col)["Churn"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    segment_churn[col] = {idx: {"churn_rate": float(row["mean"]), "n": int(row["count"])} for idx, row in seg.iterrows()}

print("\n=== Churn rate by Contract ===")
for k, v in segment_churn["Contract"].items():
    print(f"  {k}: {v['churn_rate']:.2%} ({v['n']} clients)")

# Churn rate по tenure buckets
df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6m", "6-12m", "12-24m", "24-48m", "48-72m"],
)
tenure_churn = df.groupby("tenure_bucket", observed=True)["Churn"].agg(["mean", "count"])
print("\n=== Churn rate by tenure bucket ===")
print(tenure_churn)

# ---------------------------------------------------------------------------
# 7. Сохранение результатов в JSON
# ---------------------------------------------------------------------------
output = {
    "dataset_info": {
        "name": "IBM Telco Customer Churn",
        "source": "https://github.com/IBM/telco-customer-churn-on-icp4d",
        "n_records": int(len(df)),
        "n_features": int(len(feature_cols)),
        "churn_rate": float(round(churn_rate, 4)),
        "n_churned": int(y.sum()),
        "n_retained": int(len(y) - y.sum()),
        "description": "Реальный публичный датасет IBM с данными о клиентах телеком-компании (7 032 после очистки).",
    },
    "model_cv_metrics": {
        "algorithm": "XGBoost (XGBClassifier)",
        "cv_folds": 5,
        "roc_auc_mean": float(round(cv_auc.mean(), 4)),
        "roc_auc_std": float(round(cv_auc.std(), 4)),
        "accuracy_mean": float(round(cv_acc.mean(), 4)),
        "accuracy_std": float(round(cv_acc.std(), 4)),
        "recall_mean": float(round(cv_recall.mean(), 4)),
        "precision_mean": float(round(cv_precision.mean(), 4)),
        "f1_mean": float(round(cv_f1.mean(), 4)),
    },
    "test_set_metrics": {
        "test_size": int(len(y_test)),
        "roc_auc": float(round(test_auc, 4)),
        "accuracy": float(round(test_acc, 4)),
        "recall": float(round(test_recall, 4)),
        "precision": float(round(test_precision, 4)),
        "f1": float(round(test_f1, 4)),
    },
    "feature_importance_top7": [
        {
            "name": str(row["feature"]),
            "value": float(round(row["importance_mean"], 4)),
        }
        for _, row in imp_df.head(7).iterrows()
    ],
    "segment_churn_rates": segment_churn,
    "tenure_churn": {
        str(idx): {
            "churn_rate": float(row["mean"]),
            "n": int(row["count"]),
        }
        for idx, row in tenure_churn.iterrows()
    },
}

# Сохранить в /home/z/my-project/data/processed/churn-real-results.json
Path("/home/z/my-project/data/processed").mkdir(parents=True, exist_ok=True)
with open("/home/z/my-project/data/processed/churn-real-results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved real analysis → /home/z/my-project/data/processed/churn-real-results.json")
