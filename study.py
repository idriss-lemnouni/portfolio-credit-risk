"""Evaluate calibrated credit scores on the UCI Taiwan cohort."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
    brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
TARGET = "default payment next month"
SEED = 90211
BUDGET = 0.20
DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


def load_cohort():
    path = HERE / "data" / "raw" / "uci.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=90) as response:
            payload = response.read(10_000_000)
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("Unexpected archive contents.")
        temp = path.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("UCI archive checksum mismatch.")
    with ZipFile(path) as archive:
        workbook = io.BytesIO(archive.read("default of credit card clients.xls"))
    frame = pd.read_excel(workbook, header=1, engine="xlrd")
    if (len(frame) != 30_000 or not frame.ID.is_unique or frame.isna().any().any()
            or set(frame[TARGET]) != {0, 1} or not (frame.LIMIT_BAL > 0).all()):
        raise ValueError("Unexpected cohort schema, size or data quality.")
    return frame


def features(frame, use_demographic):
    payments = frame[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
    bills = frame[[f"BILL_AMT{i}" for i in range(1, 7)]]
    data = frame.drop(columns=["ID", TARGET, *DEMOGRAPHIC])
    data = data.assign(
        delayed_months=payments.gt(0).sum(axis=1),
        maximum_delay=payments.max(axis=1),
        recent_utilization=frame.BILL_AMT1 / frame.LIMIT_BAL,
        mean_utilization=bills.mean(axis=1) / frame.LIMIT_BAL,
    )
    if use_demographic:
        demographic = frame[DEMOGRAPHIC].copy()
        demographic.EDUCATION = demographic.EDUCATION.where(demographic.EDUCATION.isin([1, 2, 3, 4]), 0)
        demographic.MARRIAGE = demographic.MARRIAGE.where(demographic.MARRIAGE.isin([1, 2, 3]), 0)
        return pd.concat([
            data, pd.get_dummies(demographic, columns=DEMOGRAPHIC[:3], dtype=float),
        ], axis=1)
    return data


def split(y):
    remainder, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    train, calibration = train_test_split(
        remainder, test_size=0.25, stratify=y[remainder], random_state=SEED + 1,
    )
    return train, calibration, test


def calibration_input(scores):
    probabilities = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def summary(y, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=(0, 1)).ravel()
    return {
        "size": len(y), "positives": int(tp + fn), "prevalence": float(y.mean()),
        "mean_probability": float(scores.mean()),
        "review_rate": float((tp + fp) / len(y)),
        "true_positive_rate": float(tp / (tp + fn)) if tp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "brier": float(brier_score_loss(y, scores)),
    }


def charts(y, scores, selected, reports, groups, model, x):
    import shap

    figures = HERE / "reports" / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#f8faf9"})

    def save(fig, name):
        fig.text(0.015, 0.012, "Source: Yeh & Lien · UCI Taiwan default cohort · locked test split", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(figures / f"{name}.png", dpi=165)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for label, probability in scores.items():
        if label == "prevalence":
            continue
        RocCurveDisplay.from_predictions(y, probability, name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, probability, name=label, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].axhline(y.mean(), color="grey", linestyle="--", label="Prevalence")
    axes[1].legend(fontsize=8)
    axes[0].set_title("Discrimination · ROC")
    axes[1].set_title("Ranking under class imbalance")
    save(fig, "discrimination")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, probability in scores.items():
        CalibrationDisplay.from_predictions(y, probability, n_bins=10, strategy="quantile", name=label, ax=ax)
    ax.set_title("Reliability on held-out accounts")
    save(fig, "calibration")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bins = np.linspace(0, 1, 26)
    for outcome, label in ((0, "No recorded default"), (1, "Default next month")):
        ax.hist(scores[selected][y == outcome], bins=bins, density=True, alpha=0.55, label=label)
    ax.axvline(reports[selected]["threshold"], color="#1c2629", linestyle="--", label="Validation cutoff")
    ax.set(xlabel="Calibrated probability", ylabel="Density", title="Predicted risk by observed outcome")
    ax.legend()
    save(fig, "risk-distribution")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, scores[selected] >= reports[selected]["threshold"], ax=ax,
        display_labels=["No default", "Default"], colorbar=False, cmap="Blues",
    )
    ax.set_title("Validation-set review rule on the test split")
    save(fig, "review-confusion")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, tpr, fpr = [], [], []
    for row in groups:
        if row["model"] == selected:
            labels.append(f"{row['group']} (n={row['size']})")
            tpr.append(row["true_positive_rate"])
            fpr.append(row["false_positive_rate"])
    pos = np.arange(len(labels))
    ax.barh(pos - 0.18, tpr, 0.36, label="Recall of defaults", color="#27776f")
    ax.barh(pos + 0.18, fpr, 0.36, label="False-positive rate", color="#bd694d")
    ax.set(yticks=pos, yticklabels=labels, xlabel="Rate", title="Error rates across recorded groups")
    ax.legend()
    save(fig, "subgroup-errors")

    sample = x.sample(n=min(240, len(x)), random_state=SEED)
    explanation = shap.TreeExplainer(model)(sample)
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    fig = plt.gcf()
    plt.title("SHAP for non-demographic boosting score", fontsize=13)
    save(fig, "boosting-shap")


def run():
    cohort = load_cohort()
    y = cohort[TARGET].to_numpy()
    train, validation, test = split(y)
    x = features(cohort, False)
    all_features = features(cohort, True)
    candidates = {
        "prevalence": (DummyClassifier(strategy="prior"), x),
        "logistic": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), x),
        "boosting": (HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=15, l2_regularization=2, random_state=SEED, early_stopping=False,
        ), x),
        "boosting + demographics": (HistGradientBoostingClassifier(
            max_iter=160, max_leaf_nodes=15, l2_regularization=2, random_state=SEED, early_stopping=False,
        ), all_features),
    }
    scores, reports, group_rows = {}, {}, []
    for label, (model, design) in candidates.items():
        model.fit(design.iloc[train], y[train])
        raw_validation = model.predict_proba(design.iloc[validation])[:, 1]
        validation_scores = raw_validation
        scores[label] = model.predict_proba(design.iloc[test])[:, 1]
        if label != "prevalence":
            calibrator = LogisticRegression().fit(calibration_input(raw_validation), y[validation])
            validation_scores = calibrator.predict_proba(calibration_input(raw_validation))[:, 1]
            scores[label] = calibrator.predict_proba(calibration_input(scores[label]))[:, 1]
        threshold = float(np.quantile(validation_scores, 1 - BUDGET, method="higher"))
        p = scores[label]
        reports[label] = summary(y[test], p, threshold) | {
            "threshold": threshold, "validation_budget": BUDGET,
            "validation_review_rate": float((validation_scores >= threshold).mean()),
            "roc_auc": float(roc_auc_score(y[test], p)),
            "average_precision": float(average_precision_score(y[test], p)),
        }
        holdout = cohort.iloc[test]
        for group, mask in {
            "Sex code 1": holdout.SEX == 1, "Sex code 2": holdout.SEX == 2,
            "Age 21–29": holdout.AGE < 30, "Age 30–49": holdout.AGE.between(30, 49),
            "Age 50+": holdout.AGE >= 50,
        }.items():
            group_rows.append(summary(y[test][mask], p[mask], threshold) | {"group": group, "model": label})
    report = {
        "dataset_sha256": SHA256, "rows": len(cohort), "split_counts": [len(train), len(validation), len(test)],
        "selected_model": "boosting", "models": reports, "groups": group_rows,
    }
    output = HERE / "reports"
    output.mkdir(exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    charts(y[test], scores, "boosting", reports, group_rows, candidates["boosting"][0], x.iloc[test])
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
