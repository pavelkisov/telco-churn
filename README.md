# Telco Customer Churn — Real Analysis

Реальный анализ оттока клиентов на публичном датасете **IBM Telco Customer Churn**.
Без синтетики, без выдуманных метрик — всё считается здесь на машине.

## Датасет

- **Источник:** https://github.com/IBM/telco-customer-churn-on-icp4d (публичный, без API-ключа Kaggle)
- **Размер:** 7 043 сырых записи × 21 признак
- **После очистки:** 7 032 записи (удалены 11 строк с пустым `TotalCharges`)
- **Целевая переменная:** `Churn` (Yes/No — ушёл клиент или остался), доля оттока 26.6%

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
| ROC-AUC (CV mean, 5-fold) | **0.8402** ± 0.0037 (округлённо **0.84**) |
| ROC-AUC (test, hold-out 20%) | **0.825** (округлённо **0.83**) |
| Accuracy (test) | 0.73 |
| Recall (churn=Yes, test) | 0.75 |
| Baseline: Logistic Regression (CV) | 0.8449 |
| Baseline: Random Forest (CV) | 0.8252 |

Числа в скобках — то же значение в двух знаках, как оно фигурирует на
сайте-портфолио: **0.84 CV / 0.83 test**.

**Top-3 признака (permutation importance):** `ContractMonths`, `tenure`, `InternetService`

> Замечание про baseline: Logistic Regression (0.8449) сопоставим с XGBoost (0.8402) —
> на этом датасете линейная модель уже близка к потолку качества, а бустинг выигрывает
> в основном в recall по классу churn (0.75 на test). Для бизнеса важнее не выжать
> +0.005 AUC, а поймать больше уходящих клиентов.

Полные результаты — в `results/churn-real-results.json`.

## Ноутбук с графиками

[`churn_eda.ipynb`](churn_eda.ipynb) — разведочный анализ и сегментация на тех же данных:

- распределение churn (дисбаланс 1:2.7 — почему ROC-AUC, а не accuracy);
- churn rate по типу контракта: month-to-month **42.7%**, годовой 11.3%, двухлетний 2.8%;
- churn rate по стажу (когорты tenure, как в `results/churn-real-results.json`): 0-6 мес **53.3%**, 6-12 мес 35.9%, 12-24 мес 28.7%, клиенты 4+ лет — **9.5%**;
- fiber-optic 41.9% против 19.0% на DSL; fiber без OnlineSecurity — 49.4%.

Моделирование — в `churn_analysis.py`.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python churn_analysis.py     # моделирование
jupyter notebook churn_eda.ipynb   # EDA (опционально)
```

Скрипт читает данные из `data/telco-churn.csv` и сохраняет результаты в `results/churn-real-results.json`
(оба пути относительные — код запускается из любого каталога после `git clone`).

## Стек

- Python 3.10+
- pandas, numpy
- scikit-learn (ColumnTransformer, Pipeline, StratifiedKFold, permutation_importance)
- xgboost
- matplotlib (ноутбук)

## Структура

```
telco-churn/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── churn_analysis.py
├── churn_eda.ipynb                 # EDA-ноутбук с графиками
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
