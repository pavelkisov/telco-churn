# Telco Customer Churn — Real Analysis

Реальный анализ оттока клиентов на публичном датасете **IBM Telco Customer Churn**.
Без синтетики, без выдуманных метрик — всё считается здесь на машине.

## Датасет

- **Источник:** https://github.com/IBM/telco-customer-churn-on-icp4d (публичный, без API-ключа Kaggle)
- **Размер:** 7 043 сырых записи × 21 признак
- **После очистки:** 7 032 записи (удалены 11 строк с пустым `TotalCharges`)
- **Целевая переменная:** `Churn` (Yes/No — ушёл клиент или остался)

## Что считается

- EDA: распределение churn, churn rate по сегментам (контракт, интернет-сервис, способ оплаты)
- Модель **XGBoost** с `scale_pos_weight` (балансировка классов) и stratified 5-fold кросс-валидацией
- **Permutation importance** — какие признаки реально влияют на отток
- Сравнение с baseline: Logistic Regression, Random Forest (тот же CV для честности)
- Feature engineering: `AvgMonthlyCharges`, `ContractMonths`, `ExtraServicesCount`
- Бизнес-инсайты: churn rate по сегментам и по когортам tenure

## Результаты (фактические, из свежего прогона)

| Метрика | Значение |
|---|---|
| ROC-AUC (CV mean, 5-fold) | **0.8402** ± std |
| ROC-AUC (test, hold-out 20%) | **0.825** |
| Accuracy (test) | 0.73 |
| Recall (churn=Yes, test) | 0.75 |
| Baseline: Logistic Regression (CV) | 0.8449 |
| Baseline: Random Forest (CV) | 0.8252 |

**Top-3 признака (permutation importance):** `ContractMonths`, `tenure`, `InternetService`

Полные результаты — в `results/churn-real-results.json`.

> Метрики получены на версии xgboost из `requirements.txt`. Из-за разницы версий библиотек
> числа могут незначительно отличаться от более ранних коммитов; это ожидаемо и
> фиксируется повторным запуском скрипта.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python churn_analysis.py
```

Скрипт читает данные из `data/telco-churn.csv` и сохраняет результаты в `results/churn-real-results.json`
(оба пути относительные — код запускается из любого каталога после `git clone`).

## Стек

- Python 3.10+
- pandas, numpy
- scikit-learn (ColumnTransformer, Pipeline, StratifiedKFold, permutation_importance)
- xgboost

## Структура

```
telco-churn/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── churn_analysis.py
├── .github/workflows/ci.yml        # CI: проверяет, что скрипт запускается
├── data/
│   └── telco-churn.csv             # исходный датасет IBM (970 КБ)
└── results/
    └── churn-real-results.json     # готовые метрики (перегенерируется запуском)
```

## Что я бы сделал дальше

- Калибровка вероятностей (CalibratedClassifierCV) — ROC-AUC высокий, но precision на churn=Yes ~0.50
- SHAP-значения вместо permutation importance для более богатой интерпретации
- Анализ retention-кривой по сегментам контракта как отдельных когорт
- Подбор порога классификации под конкретную стоимость ложноположительного/ложноотрицательного
