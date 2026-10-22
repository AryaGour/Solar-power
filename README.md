# Solar Power Generation Prediction Using Machine Learning with Weather-Based Feature Analysis

A supervised-regression project that estimates **plant-level AC power output (`AC_POWER`, kW)** of a solar plant from measured weather and time features. It follows the **Knowledge Discovery in Databases (KDD)** workflow and compares three regression models using a **chronological 80:20 split**.

> **Important:** this is **same-timestamp estimation using measured weather**, not future forecasting. Results come from one plant and about 34 days of data and must not be taken as evidence of year-round forecasting ability.

---

## Table of contents

1. [Project summary](#project-summary)
2. [Dataset](#dataset)
3. [Workflow (KDD)](#workflow-kdd)
4. [Features and leakage prevention](#features-and-leakage-prevention)
5. [Models](#models)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [How to run](#how-to-run)
9. [Repository contents](#repository-contents)
10. [Limitations](#limitations)
11. [References](#references)
12. [Author and licence](#author-and-licence)

---

## Project summary

| Item | Details |
| --- | --- |
| Domain | Renewable energy |
| Task | Supervised regression |
| Target | `AC_POWER` (kW), summed over inverters at each timestamp |
| Inputs | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `HOUR`, `MINUTE`, `DAY_OF_WEEK` |
| Models | Linear Regression, Random Forest Regressor, Gradient Boosting Regressor |
| Split | Chronological 80% train / 20% test, no shuffling |
| Metrics | MAE, RMSE, R² |
| Tools | Python, Pandas, NumPy, Matplotlib, scikit-learn, kagglehub (Google Colab / Jupyter) |

## Dataset

- **Source:** *Indian Solar Power Generation Data* on Kaggle (`anikannal/solar-power-generation-data`).
- **Scope:** Plant 1 only, using `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv`.
- **Size:** generation file 68,778 rows × 7 columns; weather file 3,182 rows × 6 columns.
- **Period:** 15 May 2020 00:00 to 17 June 2020 23:45, readings every 15 minutes (about 34 days).
- **Quality checks:** no missing values and no duplicate rows in either file.
- **Download:** automatic via `kagglehub`; no manual download is required (internet access needed).

| File | Columns |
| --- | --- |
| Generation | `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` |
| Weather | `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` |

## Workflow (KDD)

1. **Data selection:** download the dataset and load the two Plant 1 files.
2. **Data understanding:** inspect columns, types, missing values and duplicates; exploratory plots.
3. **Pre-processing:** parse timestamps; sum inverter `AC_POWER` per timestamp (68,778 rows → 3,158 plant-level timestamps); average weather per timestamp; inner-join generation and weather on `PLANT_ID` + `DATE_TIME` (3,157 merged rows).
4. **Transformation:** derive `HOUR`, `MINUTE`, `DAY_OF_WEEK`; define the feature set.
5. **Modelling:** train three regressors in scikit-learn pipelines.
6. **Evaluation:** chronological split; MAE, RMSE, R² on the test set.
7. **Interpretation:** model comparison, actual-vs-predicted plot, residual analysis, feature importance.

## Features and leakage prevention

- Inputs: three weather variables plus three time features derived from `DATE_TIME`.
- **Excluded from inputs:** `DC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`, `SOURCE_KEY`, to reduce data leakage.
- Imputation and scaling sit inside scikit-learn pipelines, so they are fitted on training data only.

## Models

| Model | Pipeline and settings |
| --- | --- |
| Linear Regression | Median imputer → `StandardScaler` → `LinearRegression` |
| Random Forest | Median imputer → `RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)` |
| Gradient Boosting | Median imputer → `GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)` |

No hyperparameter tuning is performed. `RANDOM_STATE = 42`.

## Evaluation

- **Training set:** 2,525 rows, 2020-05-15 00:00 to 2020-06-11 09:15.
- **Test set:** 632 rows, 2020-06-11 09:30 to 2020-06-17 23:45.
- **MAE** = mean(|y − ŷ|) · **RMSE** = √mean((y − ŷ)²) · **R²** = 1 − SS_res / SS_tot.
- The best model is the one with the lowest **test RMSE**.

## Results

Test-set results from the executed notebook:

| Model | MAE (kW) | RMSE (kW) | R² |
| --- | --- | --- | --- |
| **Gradient Boosting** (best by RMSE) | 244.8238 | 480.0973 | 0.9964 |
| Random Forest | 238.9281 | 490.4442 | 0.9962 |
| Linear Regression | 502.6576 | 806.3010 | 0.9898 |

- Gradient Boosting has the lowest RMSE; Random Forest has the lowest MAE. The two ensembles are close, and the notebook uses a single run, so their ordering is tentative.
- **Feature importance (Gradient Boosting):** `IRRADIATION` 0.998540, `HOUR` 0.000511, `MODULE_TEMPERATURE` 0.000425, `AMBIENT_TEMPERATURE` 0.000346, `DAY_OF_WEEK` 0.000166, `MINUTE` 0.000012. Importance shows predictive usefulness, not causation.

## How to run

1. Open `solar_power_generation_prediction_auto_download.ipynb` in Google Colab or Jupyter.
2. Make sure the environment has internet access (the dataset is downloaded with `kagglehub`).
3. Run all cells in order (**Runtime → Run all** in Colab).

**Requirements:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `kagglehub`. The notebook installs `kagglehub` itself with `%pip install kagglehub`; the other libraries are preinstalled in Colab.

```bash
pip install pandas numpy matplotlib scikit-learn kagglehub jupyter
jupyter notebook solar_power_generation_prediction_auto_download.ipynb
```

## Repository contents

```
.
├── solar_power_generation_prediction_auto_download.ipynb   # executed notebook
├── README.md                                               # this file
├── report/                                                 # project report (add your exported file)
└── presentation/                                           # slide deck (add your exported file)
```

Adjust the folder names to match your repository.

## Limitations

- **One plant** (Plant 1) only.
- **About 34 days** of data (May–June 2020); no other seasons.
- **Same-timestamp estimation:** the models use weather measured at the same moment, so they cannot forecast future output.
- Single run and single test period (about 6.6 days); no cross-validation, confidence intervals or training-set metrics.
- The number of inverters reporting at each timestamp was not checked before summing.
- The final "illustrative prediction" cell uses example inputs, not dataset measurements.

## References

- Kannal, A. (n.d.). *Solar power generation data* [Data set]. Kaggle. https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine, 17*(3), 37–54.

## Author and licence

- **Author:** [Student Name] · Roll No. [Roll Number]
- **Department / Institution:** [Department], [Institution]
- **Academic year:** [20XX–20XX]
- **Licence:** [Add a licence, e.g. MIT, if you publish this repository]. The dataset remains subject to its own Kaggle licence terms.
