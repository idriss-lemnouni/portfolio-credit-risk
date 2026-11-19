"""Inspect aggregate benchmark results without entering applicant data."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


REPORT = Path(__file__).resolve().parent / "reports"
st.set_page_config(page_title="Credit default score audit", layout="wide")
st.title("Credit default score audit")
st.caption("UCI Taiwan cohort · October 2005 outcomes · 6,000 held-out accounts")
report = json.loads((REPORT / "metrics.json").read_text(encoding="utf-8"))
name = st.selectbox("Model", [n for n in report["models"] if n != "prevalence"], index=1)
metrics = report["models"][name]
columns = st.columns(4)
for column, key, value in zip(
    columns, ("ROC AUC", "Average precision", "Brier score", "Test review rate"),
    (f"{metrics['roc_auc']:.3f}", f"{metrics['average_precision']:.3f}",
     f"{metrics['brier']:.3f}", f"{metrics['review_rate']:.1%}"), strict=True,
):
    column.metric(key, value)
st.subheader("Error and calibration audit by group")
cohort = st.radio("Group definition", ["Recorded sex", "Age bracket"], horizontal=True)
prefix = {"Recorded sex": "Sex", "Age bracket": "Age"}[cohort]
rows = [row for row in report["groups"] if row["model"] == name and row["group"].startswith(prefix)]
frame = pd.DataFrame(rows).set_index("group")
st.bar_chart(frame[["true_positive_rate", "false_positive_rate"]])
st.dataframe(frame[["size", "positives", "prevalence", "mean_probability", "review_rate", "brier"]])
st.download_button("Download aggregate metrics", (REPORT / "metrics.json").read_bytes(), "credit-risk-metrics.json")
st.caption("Historical benchmark for model evaluation. Review-rate thresholds were fixed on the calibration split.")
