# ⚡ Load Shedding Prediction Using Machine Learning & Explainable AI

An end-to-end machine learning project for predicting electricity load-shedding risk using grid demand, generation, weather, market, and operational conditions.

The project includes data validation, leakage prevention, exploratory analysis, time-based model evaluation, class-imbalance handling, threshold analysis, explainable AI using SHAP, model persistence, and an interactive Streamlit application.

> **Dataset Notice:** This project uses synthetic hourly electricity-grid data generated for educational and portfolio purposes. It should not be represented as real operational data from a German transmission or distribution system operator.

---

## 🎯 Project Objective

Electricity grids must continuously balance demand and generation.

During periods of:

* high electricity demand,
* insufficient reserve capacity,
* stressed generation conditions,
* abnormal operational conditions,

the risk of load shedding may increase.

The objective of this project is to build a machine learning system that estimates whether load shedding is likely under given grid conditions.

Target variable:

```text
load_shed_flag

0 = No Load Shedding
1 = Load Shedding
```

---

## 📊 Dataset

The dataset contains hourly observations covering:

```text
2020-01-01 to 2024-12-31
```

Original dataset size:

```text
43,848 rows
38 columns
```

After data cleaning:

```text
43,680 rows
36 columns
```

The target is highly imbalanced:

```text
No Load Shedding: ~93.6%
Load Shedding:    ~6.4%
```

This makes accuracy alone unsuitable for evaluating the models.

Therefore, the project also evaluates:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall performance
* Confusion Matrix

---

## 🔍 Data Leakage Prevention

Two columns were removed before model training:

```text
load_shed_amount_mw
load_shed_priority_tier
```

These variables contain information directly associated with the load-shedding outcome and could cause target leakage.

Removing them ensures a more realistic prediction pipeline.

---

## 🧹 Data Processing

The preprocessing pipeline includes:

* timestamp conversion,
* missing-value analysis,
* duplicate validation,
* chronological sorting,
* lag-feature missing-value removal,
* target-leakage removal,
* categorical encoding,
* feature preparation,
* chronological train-test splitting.

The final machine-learning dataset contains:

```text
44 model features
```

---

## ⏱️ Time-Based Train-Test Split

Because the dataset is chronological electricity-grid data, random shuffling was avoided.

The data was split chronologically:

```text
Training samples: 34,944
Testing samples:   8,736
```

This provides a more realistic evaluation because the model learns from historical observations and predicts later observations.

---

## 📈 Exploratory Data Analysis

EDA showed a strong relationship between grid conditions and load shedding.

### Average Reserve Margin

```text
No Load Shedding: 46.39%
Load Shedding:    25.39%
```

### Average Electricity Demand

```text
No Load Shedding: 590.94 MW
Load Shedding:    817.16 MW
```

This indicates that load-shedding events in the synthetic dataset are associated with:

* higher electricity demand,
* lower reserve margins.

Generated EDA plots are stored in:

```text
reports/
```

---

## 🤖 Machine Learning Models

Four approaches were evaluated:

1. Dummy Baseline Classifier
2. Logistic Regression
3. Random Forest
4. XGBoost

Class imbalance was handled using class weighting or positive-class weighting where appropriate.

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall |   F1 Score |    ROC-AUC |
| ------------------- | -------: | --------: | -----: | ---------: | ---------: |
| Baseline            |   93.43% |     0.00% |  0.00% |      0.00% |        N/A |
| Logistic Regression |   81.63% |    24.11% | 83.62% |     37.43% |     86.39% |
| Random Forest       |   90.91% |    40.28% | 79.44% | **53.46%** | **87.50%** |
| XGBoost             |   91.23% |    40.75% | 73.69% |     52.48% |     86.76% |

Although the baseline achieves high accuracy, it predicts only the majority class and fails to identify any load-shedding events.

Random Forest achieved the strongest overall F1 score and ROC-AUC among the trained models.

---

## 🏆 Final Model

The selected model is:

```text
Random Forest Classifier
```

Performance on the chronological test set:

```text
Accuracy  : 90.91%
Precision : 40.28%
Recall    : 79.44%
F1 Score  : 53.46%
ROC-AUC   : 87.50%
```

Confusion matrix:

```text
[[7486  676]
 [ 118  456]]
```

Interpretation:

```text
7486 = Correct No-Load-Shedding predictions
676  = False alarms
118  = Missed load-shedding events
456  = Correct load-shedding detections
```

---

## 🎯 Classification Threshold Analysis

The default classification threshold was:

```text
0.50
```

Threshold optimization was performed using a separate chronological validation segment rather than tuning directly on the test set.

Validation-selected threshold:

```text
0.6038
```

Validation PR performance:

```text
Average Precision / PR-AUC: 0.3381
```

Test performance at the validation-selected threshold:

```text
Accuracy  : 91.35%
Precision : 41.03%
Recall    : 72.47%
F1 Score  : 52.39%
```

The higher threshold reduced false alarms but increased missed load-shedding events.

Because the default `0.50` threshold achieved better test F1 and higher recall, the final deployment configuration retains:

```text
Classification Threshold = 0.50
```

---

## 🧠 Explainable AI

The project includes two explainability approaches:

### Random Forest Feature Importance

Top features include:

| Rank | Feature                    | Importance |
| ---: | -------------------------- | ---------: |
|    1 | reserve_margin_pct         |     0.2124 |
|    2 | total_demand_mw            |     0.1455 |
|    3 | residential_load_mw        |     0.1135 |
|    4 | conventional_generation_mw |     0.0869 |
|    5 | demand_lag_168h            |     0.0833 |

### SHAP

SHAP analysis was implemented to examine how individual features influence model predictions.

Generated explainability plots:

```text
reports/random_forest_feature_importance.png
reports/shap_feature_importance.png
reports/shap_summary.png
```

Feature importance identifies which variables are influential, while SHAP provides additional information about how feature values contribute to predictions.

---

## 🖥️ Streamlit Application

An interactive Streamlit dashboard allows users to enter:

* date and time information,
* weather conditions,
* industrial demand,
* residential demand,
* commercial demand,
* lagged demand,
* renewable generation,
* conventional generation,
* grid frequency,
* reserve margin,
* fuel prices,
* electricity prices,
* feeder information,
* criticality score.

The application returns:

```text
Load Shedding Probability
Prediction
Risk Level
Classification Threshold
```

---

## 📁 Project Structure

```text
load_shedding_prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── final_random_forest.pkl
│   ├── feature_columns.pkl
│   └── classification_threshold.pkl
│
├── notebooks/
│
├── reports/
│   ├── target_distribution.png
│   ├── demand_distribution.png
│   ├── reserve_margin_vs_load_shedding.png
│   ├── demand_vs_load_shedding.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── random_forest_precision_recall_curve.png
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_feature_importance.png
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   ├── model_comparison.py
│   ├── threshold_optimization.py
│   ├── explainability.py
│   ├── model_saving.py
│   └── prediction.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SRP500005/Load-Shedding-Prediction.git
```

Move into the project:

```bash
cd Load-Shedding-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Machine Learning Pipeline

Run:

```bash
python main.py
```

The complete pipeline performs:

```text
Data Loading
      ↓
Data Validation
      ↓
Data Cleaning
      ↓
Leakage Prevention
      ↓
EDA
      ↓
Feature Engineering
      ↓
Time-Based Split
      ↓
Baseline Model
      ↓
Logistic Regression
      ↓
Random Forest
      ↓
XGBoost
      ↓
Model Comparison
      ↓
Threshold Analysis
      ↓
SHAP Explainability
      ↓
Final Model Saving
```

---

## 🌐 Run Streamlit Application

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Joblib
* Streamlit
* Git
* GitHub

---

## 🔮 Future Improvements

Potential extensions include:

* training on real grid or market data,
* TimeSeriesSplit cross-validation,
* hyperparameter optimization,
* probability calibration,
* cost-sensitive decision thresholds,
* real-time grid-data integration,
* API deployment,
* cloud deployment,
* model monitoring,
* drift detection,
* automated retraining.

---

## ⚠️ Disclaimer

The dataset used in this repository is synthetic.

The project demonstrates machine-learning methodology and software-engineering practices and should not be interpreted as a production-ready electricity-grid control system.

Any operational power-system deployment would require validated real-world data, domain-specific safety analysis, regulatory review, and extensive testing.

---

## 👨‍💻 Project Purpose

This project was developed as an end-to-end machine-learning portfolio project demonstrating:

* data validation,
* data leakage prevention,
* imbalanced classification,
* time-aware evaluation,
* model comparison,
* explainable AI,
* reusable inference,
* model persistence,
* and interactive deployment.
