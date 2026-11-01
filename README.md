# Telco Customer Churn — Real Analysis

Реальный анализ оттока клиентов на публичном датасете **IBM Telco Customer Churn**.
Без синтетики, без выдуманных метрик — всё считается здесь на машине.

## Датасет

- **Источник:** https://github.com/IBM/telco-customer-churn-on-icp4d (публичный, без API-ключа)
- **Размер:** 7 043 записи × 21 признак
- **Целевая переменная:** `Churn` (Yes/No — ушёл клиент или остался)

## Что считается

- EDA: распределение churn, churn rate по сегментам (контракт, способ оплаты, интернет-сервис)
- Модель **XGBoost** с stratified 5-fold кросс-валидацией
- **Permutation importance** — какие признаки реально влияют на отток
- Сравнение с baseline: Logistic Regression, Random Forest
- Бизнес-инсайты: какие сегменты клиентов нужно удерживать в первую очередь

## Результаты (фактические)

- **ROC-AUC (CV mean):** 0.8402
- **ROC-AUC (test):** 0.825
- **Accuracy:** 0.79
- **Recall (churn=Yes):** 0.51
- **Top-3 признака:** Contract (Month-to-month), tenure, OnlineSecurity_No

Полные результаты — в `results/churn-real-results.json`.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python churn_analysis.py
```

В конце скрипт выведет метрики в консоль и сохранит JSON в `results/`.

## Стек

- Python 3.10+
- pandas, numpy
- scikit-learn (ColumnTransformer, Pipeline, StratifiedKFold)
- XGBoost

## Структура

```
telco-churn/
├── README.md
├── requirements.txt
├── churn_analysis.py
├── data/
│   └── telco-churn.csv          # исходный датасет IBM (948 KB)
└── results/
    └── churn-real-results.json  # готовые метрики (для сверки)
```
