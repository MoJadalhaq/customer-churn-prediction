"""
Customer Churn Prediction Dashboard
====================================
Upload a CSV dataset → train a Random Forest → explore churn predictions,
feature importances, SHAP explanations, and retention recommendations.
"""

import io
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# Make sure local packages resolve regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.train_model import train_model  # noqa: E402
from utils.recommend import get_recommendations  # noqa: E402


def _prob_gradient(col, low_color="#ffffff", high_color="#e74c3c"):
    """Return inline CSS background colours scaled between two hex colours."""
    lo = tuple(int(low_color[i:i+2], 16) for i in (1, 3, 5))
    hi = tuple(int(high_color[i:i+2], 16) for i in (1, 3, 5))
    styles = []
    vmin, vmax = col.min(), col.max()
    span = vmax - vmin if vmax != vmin else 1.0
    for v in col:
        t = (v - vmin) / span
        r, g, b = (int(lo[i] + t * (hi[i] - lo[i])) for i in range(3))
        styles.append(f"background-color: rgb({r},{g},{b})")
    return styles

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Prediction Dashboard")
st.caption(
    "Upload a customer dataset · train a Random Forest · get churn predictions, "
    "SHAP explanations, and retention recommendations."
)

# ── Sidebar — file upload & controls ──────────────────────────────────────────
_SAMPLE_PATH = pathlib.Path(__file__).parent / "data" / "sample_bank_churn.csv"

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    st.caption("— or —")
    load_sample_btn = st.button(
        "📂 Load Sample Dataset",
        use_container_width=True,
        help="Loads the built-in 100-row Thera Bank sample so you can explore the dashboard right away.",
    )

if load_sample_btn:
    st.session_state["use_sample"] = True
    # Clear any previous training results so the new dataset starts fresh
    for _k in ("model", "encoders", "X", "y", "report", "feature_names", "target_col"):
        st.session_state.pop(_k, None)

# ── Guard: nothing uploaded yet ────────────────────────────────────────────────
_using_sample = st.session_state.get("use_sample", False) and uploaded_file is None

if uploaded_file is None and not _using_sample:
    st.info("👈 Upload a CSV file in the sidebar to get started, or click **Load Sample Dataset** to try the built-in demo.")
    with st.expander("Expected CSV format"):
        st.markdown(
            """
- **One row per customer**
- **One column** = the churn label (`0/1`, `Yes/No`, `True/False`, etc.)
- Remaining columns are feature columns (numeric or categorical)
- Missing values are handled automatically
            """
        )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.session_state.pop("use_sample", None)  # a real upload overrides the sample
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_csv(_SAMPLE_PATH)
    st.info("Using built-in sample dataset (100 rows · Thera Bank structure). Upload your own CSV to replace it.")

with st.sidebar:
    st.success(f"✅ Loaded **{len(df_raw):,}** rows × **{len(df_raw.columns)}** cols")
    target_col = st.selectbox("🎯 Target column (churn label)", df_raw.columns.tolist())
    st.divider()
    st.subheader("🛠️ Model Parameters")
    test_size_pct = st.slider(
        "Test set size (%)",
        min_value=5, max_value=50, value=20, step=5,
        help="Percentage of data held out for evaluation",
    )
    n_estimators = st.slider(
        "Number of trees (n_estimators)",
        min_value=50, max_value=500, value=200, step=50,
        help="More trees = more stable but slower to train",
    )
    st.divider()
    train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

# ── Data preview ──────────────────────────────────────────────────────────────
st.subheader("📋 Dataset Preview")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Customers", f"{len(df_raw):,}")
col_b.metric("Feature Columns", len(df_raw.columns) - 1)

if target_col:
    raw_vals = df_raw[target_col].astype(str).str.strip().str.lower()
    churn_rate = raw_vals.isin(["1", "yes", "true"]).mean()
    col_c.metric("Observed Churn Rate", f"{churn_rate:.1%}")

st.dataframe(df_raw.head(10), use_container_width=True)

# ── Train model ───────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training Random Forest … this may take a few seconds."):
        try:
            model, encoders, X, y, report, feature_names, y_test, y_pred_test, y_prob_test = train_model(
                df_raw, target_col,
                test_size=test_size_pct / 100,
                n_estimators=n_estimators,
            )
            st.session_state.update(
                {
                    "model": model,
                    "encoders": encoders,
                    "X": X,
                    "y": y,
                    "report": report,
                    "feature_names": feature_names,
                    "target_col": target_col,
                    "y_test": y_test,
                    "y_pred_test": y_pred_test,
                    "y_prob_test": y_prob_test,
                    "test_size_pct": test_size_pct,
                    "n_estimators": n_estimators,
                }
            )
            st.success("✅ Model trained successfully!")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

if "model" not in st.session_state:
    st.info("👈 Click **Train Model** in the sidebar to continue.")
    st.stop()

# ── Restore session ────────────────────────────────────────────────────────────
model        = st.session_state["model"]
X            = st.session_state["X"]
y            = st.session_state["y"]
report       = st.session_state["report"]
feature_names= st.session_state["feature_names"]
y_test       = st.session_state["y_test"]
y_pred_test  = st.session_state["y_pred_test"]
y_prob_test  = st.session_state["y_prob_test"]
_test_pct    = st.session_state.get("test_size_pct", 20)
_n_est       = st.session_state.get("n_estimators", 200)

# ── Model performance ─────────────────────────────────────────────────────────
st.divider()
st.subheader(f"📊 Model Performance (held-out {_test_pct}% test set · {_n_est} trees)")

churn_key = "1"  # classification_report uses str(label)
accuracy  = report.get("accuracy", 0)
precision = report.get(churn_key, {}).get("precision", 0)
recall    = report.get(churn_key, {}).get("recall", 0)
f1        = report.get(churn_key, {}).get("f1-score", 0)
support   = int(report.get(churn_key, {}).get("support", 0))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy",        f"{accuracy:.1%}")
m2.metric("Precision",       f"{precision:.1%}", help="Of predicted churners, how many actually churned")
m3.metric("Recall",          f"{recall:.1%}",    help="Of actual churners, how many were caught")
m4.metric("F1 Score",        f"{f1:.1%}")
m5.metric("Test Churners",   f"{support:,}")

# Confusion matrix + ROC curve side by side
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc  # noqa: E402

_cm = confusion_matrix(y_test, y_pred_test)
_cm_labels = ["No Churn", "Churn"]

fig_cm = go.Figure(
    go.Heatmap(
        z=_cm,
        x=_cm_labels,
        y=_cm_labels,
        colorscale="Blues",
        showscale=False,
        text=_cm,
        texttemplate="%{text}",
        textfont={"size": 18},
    )
)
fig_cm.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted",
    yaxis_title="Actual",
    yaxis_autorange="reversed",
    height=340,
)

_fpr, _tpr, _ = roc_curve(y_test, y_prob_test)
_auc_score = sk_auc(_fpr, _tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=_fpr, y=_tpr,
    mode="lines",
    name=f"AUC = {_auc_score:.3f}",
    line=dict(color="#3498db", width=2),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Random baseline",
    line=dict(color="#bdc3c7", width=1, dash="dash"),
))
fig_roc.update_layout(
    title=f"ROC Curve  ·  AUC = {_auc_score:.3f}",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.6, y=0.1),
    height=340,
)

col_cm, col_roc = st.columns(2)
with col_cm:
    st.plotly_chart(fig_cm, use_container_width=True)
with col_roc:
    st.plotly_chart(fig_roc, use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.subheader("🔑 Feature Importance")

fi_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=True)
    .tail(20)
)

fig_fi = px.bar(
    fi_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top-20 Feature Importances (Random Forest)",
    color="Importance",
    color_continuous_scale="Blues",
    labels={"Importance": "Importance Score"},
)
fig_fi.update_layout(height=max(420, len(fi_df) * 26), coloraxis_showscale=False)
st.plotly_chart(fig_fi, use_container_width=True)

# ── Predictions ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔮 Churn Predictions")

probs = model.predict_proba(X)[:, 1]
preds = model.predict(X)

result_df = df_raw.copy()
result_df["Churn_Predicted"] = preds
result_df["Churn_Probability"] = probs.round(4)
result_df["Risk_Level"] = pd.cut(
    probs,
    bins=[0.0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

p1, p2, p3 = st.columns(3)
p1.metric("Predicted to Churn",     f"{int(preds.sum()):,}")
p2.metric("Predicted to Stay",      f"{int((preds == 0).sum()):,}")
p3.metric("Avg Churn Probability",  f"{probs.mean():.1%}")

# Risk distribution donut
risk_counts = result_df["Risk_Level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]
fig_risk = px.pie(
    risk_counts,
    values="Count",
    names="Risk Level",
    title="Customer Risk Distribution",
    color="Risk Level",
    color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
    hole=0.45,
)
fig_risk.update_traces(textposition="outside", textinfo="percent+label")
st.plotly_chart(fig_risk, use_container_width=True)

# ── SHAP global summary ───────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 SHAP Explanations — Global Feature Impact")

SHAP_SAMPLE = min(300, len(X))

with st.spinner(f"Computing SHAP values on a {SHAP_SAMPLE}-row sample …"):
    X_sample = X.iloc[:SHAP_SAMPLE].copy()
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_sample)

    # RandomForestClassifier returns [class0_array, class1_array]
    if isinstance(raw_shap, list) and len(raw_shap) == 2:
        shap_churn = np.array(raw_shap[1])  # class 1 = churn
    else:
        shap_churn = np.array(raw_shap)

mean_abs_shap = np.abs(shap_churn).mean(axis=0).flatten()
mean_abs_shap = mean_abs_shap[:len(feature_names)]
shap_global_df = (
    pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap})
    .sort_values("Mean |SHAP|", ascending=True)
    .tail(20)
)

fig_shap = px.bar(
    shap_global_df,
    x="Mean |SHAP|",
    y="Feature",
    orientation="h",
    title="Mean Absolute SHAP Values — Average Impact on Churn Probability",
    color="Mean |SHAP|",
    color_continuous_scale="Reds",
)
fig_shap.update_layout(height=max(420, len(shap_global_df) * 26), coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)

# ── Individual customer deep-dive ─────────────────────────────────────────────
st.divider()
st.subheader("👤 Individual Customer Deep-Dive")

cust_idx = st.slider(
    "Select Customer Index",
    min_value=0,
    max_value=SHAP_SAMPLE - 1,
    value=0,
    help="Explore SHAP explanations and recommendations for a specific customer",
)

cust_prob = float(probs[cust_idx])
cust_pred = "Will Churn" if preds[cust_idx] == 1 else "Will Retain"
risk_emoji = "🔴" if cust_prob >= 0.6 else ("🟡" if cust_prob >= 0.3 else "🟢")
risk_label = "High" if cust_prob >= 0.6 else ("Medium" if cust_prob >= 0.3 else "Low")

c1, c2, c3 = st.columns(3)
c1.metric("Prediction",       cust_pred)
c2.metric("Churn Probability", f"{cust_prob:.1%}")
c3.metric("Risk Level",        f"{risk_emoji} {risk_label}")

# SHAP waterfall bar chart for this customer
cust_shap = shap_churn[cust_idx]
cust_shap_df = (
    pd.DataFrame({"Feature": feature_names, "SHAP Value": cust_shap})
    .sort_values("SHAP Value")
)
bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in cust_shap_df["SHAP Value"]]

fig_wf = go.Figure(
    go.Bar(
        x=cust_shap_df["SHAP Value"],
        y=cust_shap_df["Feature"],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    )
)
fig_wf.update_layout(
    title=f"SHAP Values for Customer #{cust_idx}  ·  Red = pushes toward churn  ·  Green = away from churn",
    xaxis_title="SHAP Value",
    height=max(420, len(feature_names) * 24),
)
st.plotly_chart(fig_wf, use_container_width=True)

# ── Retention recommendations ─────────────────────────────────────────────────
st.subheader("💡 Retention Recommendations")

if cust_prob >= 0.3:
    recs = get_recommendations(cust_shap, feature_names, top_n=3)
    for rank, rec in enumerate(recs, start=1):
        st.markdown(f"**{rank}.** {rec}")
else:
    st.success(
        "✅ This customer shows **low churn risk**. "
        "Consider a light-touch loyalty nudge rather than a heavy intervention."
    )

# ── High Risk Customers ───────────────────────────────────────────────────────
st.divider()
st.subheader("🚨 High Risk Customers")

high_risk_df = result_df[result_df["Churn_Probability"] > 0.6].copy()
high_risk_df = high_risk_df.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)

hr1, hr2, hr3 = st.columns(3)
hr1.metric("High Risk Count",       f"{len(high_risk_df):,}")
hr2.metric("% of All Customers",    f"{len(high_risk_df) / max(len(result_df), 1):.1%}")
hr3.metric("Avg Probability (High)", f"{high_risk_df['Churn_Probability'].mean():.1%}" if len(high_risk_df) else "—")

if high_risk_df.empty:
    st.success("No customers with churn probability above 60%.")
else:
    preview_feature_cols_hr = [c for c in df_raw.columns if c != target_col][:5]
    hr_display_cols = ["Churn_Probability", "Risk_Level"] + preview_feature_cols_hr
    st.dataframe(
        high_risk_df[hr_display_cols]
        .style.apply(_prob_gradient, subset=["Churn_Probability"]),
        use_container_width=True,
        height=min(420, 40 + len(high_risk_df) * 35),
    )

    hr_csv = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download High Risk Customers CSV",
        data=hr_csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

# ── Full predictions table ────────────────────────────────────────────────────
st.divider()
st.subheader("📄 Full Predictions Table")

preview_feature_cols = [c for c in df_raw.columns if c != target_col][:5]
display_cols = ["Churn_Predicted", "Churn_Probability", "Risk_Level"] + preview_feature_cols

st.dataframe(
    result_df[display_cols]
    .style.apply(_prob_gradient, subset=["Churn_Probability"]),
    use_container_width=True,
    height=420,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Download Predictions as CSV",
    data=csv_bytes,
    file_name="churn_predictions.csv",
    mime="text/csv",
    type="primary",
)
