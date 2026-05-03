# Customer Churn Prediction Dashboard

An interactive Streamlit web app that lets you upload a customer dataset, train a Random Forest classifier in-browser, and immediately explore churn predictions, feature importance, SHAP explanations, and actionable retention recommendations — no coding required.

---

## Features

- **CSV upload** — drag-and-drop any customer dataset; the app auto-detects numeric and categorical columns
- **One-click training** — trains a 200-tree Random Forest with balanced class weights and an 80/20 stratified split
- **Model performance panel** — accuracy, precision, recall, F1, and test-set churn count at a glance
- **Feature importance chart** — top-20 features ranked by Random Forest impurity decrease (Plotly, interactive)
- **Churn predictions table** — every customer scored with a probability, a predicted label, and a Low / Medium / High risk tier
- **Risk distribution donut** — instant visual breakdown of your customer base by risk tier
- **Global SHAP explanations** — mean absolute SHAP bar chart showing average feature impact on churn probability
- **Individual customer deep-dive** — per-customer SHAP waterfall chart; red bars push toward churn, green push away
- **Retention recommendations** — up to 3 prioritised, actionable interventions derived from each customer's top churn-driving features
- **Download predictions** — export the full scored dataset as a CSV with one click

---

## Tech Stack

| Layer | Library |
|---|---|
| UI / app framework | [Streamlit](https://streamlit.io/) >= 1.28 |
| ML model | [scikit-learn](https://scikit-learn.org/) >= 1.3 (RandomForestClassifier) |
| Explainability | [SHAP](https://shap.readthedocs.io/) >= 0.43 (TreeExplainer) |
| Charts | [Plotly](https://plotly.com/python/) >= 5.17 |
| Data | [pandas](https://pandas.pydata.org/) >= 2.0, [NumPy](https://numpy.org/) >= 1.24 |
| Model persistence | [joblib](https://joblib.readthedocs.io/) >= 1.3 |

---

## Project Structure

```
CustomerChurn/
├── app.py                  # Streamlit dashboard (main entry point)
├── requirements.txt
├── model/
│   ├── __init__.py
│   └── train_model.py      # Preprocessing pipeline + RandomForest training
└── utils/
    ├── __init__.py
    └── recommend.py        # SHAP-driven retention recommendation engine
```

---

## Installation

**Prerequisites:** Python 3.8 or later.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-churn-dashboard.git
cd customer-churn-dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Usage

1. **Upload** your CSV using the sidebar file uploader.
2. **Select** the column that contains the churn label from the dropdown.
3. Click **Train Model** — training typically completes in a few seconds.
4. Explore the results across the dashboard sections: performance metrics, feature importance, predictions, SHAP explanations, and retention recommendations.
5. Use the **customer index slider** to drill into any individual customer.
6. Click **Download Predictions as CSV** to export the scored dataset.

---

## Expected CSV Format

| Requirement | Detail |
|---|---|
| Layout | One row per customer |
| Churn column | Any single column containing the churn label — the app lets you choose it at runtime |
| Accepted label values | `0` / `1`, `Yes` / `No`, `True` / `False` (case-insensitive) |
| Feature columns | All remaining columns; numeric and categorical are both supported |
| Missing values | Handled automatically — numeric columns filled with the column median, categoricals filled with `"Unknown"` |

**Minimal example:**

```
customerID,gender,tenure,MonthlyCharges,Contract,Churn
7590-VHVEG,Female,1,29.85,Month-to-month,No
5575-GNVDE,Male,34,56.95,One year,No
3668-QPYBK,Male,2,53.85,Month-to-month,Yes
```

The app works with the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) out of the box, and with any similarly structured dataset.

---

## How It Works

1. **Preprocessing** (`model/train_model.py`) — drops rows with missing target values, median-imputes numeric features, label-encodes categoricals.
2. **Training** — `RandomForestClassifier(n_estimators=200, class_weight="balanced")` fit on 80 % of the data; performance evaluated on the held-out 20 %.
3. **SHAP** — `shap.TreeExplainer` computes exact SHAP values for up to 300 sampled rows, producing both global summaries and per-customer breakdowns.
4. **Recommendations** (`utils/recommend.py`) — each customer's positive-SHAP features are ranked and matched against a keyword-to-strategy lookup table to generate targeted retention actions.

---

## License

MIT
