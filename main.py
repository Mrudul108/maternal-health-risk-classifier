"""
MHRLC: Maternal Health Risk Level Classification Using Machine Learning.

Educational mini-project for HDSSBL701. The model predicts dataset-defined
risk labels (low risk, mid risk, high risk). It is not a clinical diagnostic
system.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
TARGET_COLUMN = "RiskLevel"
CLASS_ORDER = ["low risk", "mid risk", "high risk"]

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "maternal_health_risk.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "mhrlc_model.joblib"


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/863/data.csv"


def download_uci_csv() -> pd.DataFrame:
    """Download the official UCI CSV, with an SSL fallback for campus machines."""
    try:
        from ucimlrepo import fetch_ucirepo

        maternal = fetch_ucirepo(id=863)
        return pd.concat([maternal.data.features, maternal.data.targets], axis=1)
    except Exception as exc:
        print(f"ucimlrepo download did not succeed ({type(exc).__name__}).")
        print("Trying the official UCI CSV URL instead...")

    try:
        return pd.read_csv(UCI_CSV_URL, encoding="utf-8-sig")
    except Exception as exc:
        print(f"Standard HTTPS download failed ({type(exc).__name__}: {exc}).")
        print(
            "Retrying with an unverified SSL context. This workaround is only "
            "for local certificate errors; the file still comes from UCI."
        )

    context = ssl._create_unverified_context()
    with urllib.request.urlopen(UCI_CSV_URL, context=context, timeout=60) as response:
        raw = response.read()
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")


def load_dataset() -> pd.DataFrame:
    """Load a local CSV if present; otherwise download from UCI and save it."""
    print_section("Stage 1: Dataset acquisition")

    if CSV_PATH.exists():
        print(f"Found local file: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        print("Loaded the dataset from the local CSV copy.")
        return df

    print("Local CSV not found. Downloading UCI dataset id=863...")
    df = download_uci_csv()
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved a local copy to: {CSV_PATH}")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print_section("Stage 2: Data inspection")

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nLast five records:")
    print(df.tail().to_string(index=False))

    print(f"\nShape (rows, columns): {df.shape}")
    print(f"Column names: {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count}")
    print(
        "These rows are reported, not removed automatically. "
        "Repeated records can occur in this dataset and may inflate "
        "test scores if they appear in both train and test splits."
    )

    print("\nTarget class distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False).to_string())
    print("\nTarget class proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_string())

    print("\nStatistical summary of numeric features:")
    print(df[FEATURE_COLUMNS].describe().round(3).to_string())


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Stage 3: Data preprocessing")

    cleaned = df.copy()

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.strip().str.lower()
    )
    print("Standardized RiskLevel labels (trim whitespace, lowercase).")
    print("Unique labels after standardization:")
    print(sorted(cleaned[TARGET_COLUMN].unique()))

    unexpected = set(cleaned[TARGET_COLUMN].unique()) - set(CLASS_ORDER)
    if unexpected:
        raise ValueError(f"Unexpected RiskLevel values: {unexpected}")

    print("\nChecking for implausible or suspicious values (not auto-removed):")
    checks = {
        "Age < 10 or Age > 70": (cleaned["Age"] < 10) | (cleaned["Age"] > 70),
        "SystolicBP < DiastolicBP": cleaned["SystolicBP"] < cleaned["DiastolicBP"],
        "HeartRate < 40": cleaned["HeartRate"] < 40,
        "BodyTemp < 95 or BodyTemp > 106": (cleaned["BodyTemp"] < 95)
        | (cleaned["BodyTemp"] > 106),
        "BS <= 0": cleaned["BS"] <= 0,
    }
    for label, mask in checks.items():
        count = int(mask.sum())
        print(f"  {label}: {count} row(s)")
        if count and label == "HeartRate < 40":
            print("    HeartRate values below 40:")
            print(cleaned.loc[mask, FEATURE_COLUMNS + [TARGET_COLUMN]].to_string(index=False))

    print(
        "\nFeature units used in this project "
        "(as commonly documented for UCI dataset 863):"
    )
    print("  Age: years")
    print("  SystolicBP / DiastolicBP: mmHg")
    print("  BS: blood glucose in mmol/L")
    print("  BodyTemp: Fahrenheit")
    print("  HeartRate: beats per minute")
    print(
        "\nRows with unusual values are kept so the analysis matches the "
        "public dataset. They are discussed as data-quality limitations."
    )
    return cleaned


def run_eda(df: pd.DataFrame) -> None:
    print_section("Stage 4: Exploratory data analysis")
    FIGURES_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET_COLUMN, order=CLASS_ORDER, ax=ax)
    ax.set_title("Risk category counts")
    ax.set_xlabel("RiskLevel")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_risk_counts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(column)
    fig.suptitle("Feature distributions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_feature_histograms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.boxplot(y=df[column], ax=ax)
        ax.set_title(column)
    fig.suptitle("Feature box plots")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_feature_boxplots.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=column, order=CLASS_ORDER, ax=ax)
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Features by risk category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_features_by_risk.png", dpi=150)
    plt.close(fig)

    print(f"Saved EDA figures in: {FIGURES_DIR}")
    print("\nMean feature values by risk category:")
    print(
        df.groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
        .mean()
        .reindex(CLASS_ORDER)
        .round(3)
        .to_string()
    )
    print(
        "\nObservations should be read from the printed statistics and "
        "saved plots. They are dataset patterns, not clinical rules."
    )


def split_data(df: pd.DataFrame):
    print_section("Stage 5: Train-test split")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training rows: {len(X_train)} ({100 * (1 - TEST_SIZE):.0f}%)")
    print(f"Testing rows:  {len(X_test)} ({100 * TEST_SIZE:.0f}%)")
    print(f"Random seed:   {RANDOM_STATE}")
    print("Stratified split is used so class proportions stay similar.")
    print("\nTrain class counts:")
    print(y_train.value_counts().reindex(CLASS_ORDER).to_string())
    print("\nTest class counts:")
    print(y_test.value_counts().reindex(CLASS_ORDER).to_string())
    print(
        "\nAny scaler is fitted on training data only, then applied to "
        "the test set, to avoid leaking test information into training."
    )
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy:          {metrics['Accuracy']:.4f}")
    print(f"Precision (macro): {metrics['Precision (macro)']:.4f}")
    print(f"Recall (macro):    {metrics['Recall (macro)']:.4f}")
    print(f"F1 (macro):        {metrics['F1 (macro)']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_string()
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"06_confusion_{safe_name}.png", dpi=150)
    plt.close(fig)
    return metrics


def train_and_compare(X_train, X_test, y_train, y_test):
    print_section("Stage 6 and 7: Model training and evaluation")
    print(
        "Logistic Regression uses StandardScaler because it is sensitive "
        "to feature magnitude. Tree-based models do not need scaling."
    )
    print(
        "Macro-averaged metrics treat each risk class equally, which is "
        "useful when class counts are not identical."
    )

    models = build_models()
    results = []
    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    comparison = pd.DataFrame(results)
    print("\nModel comparison (test set):")
    print(comparison.round(4).to_string(index=False))
    print(
        "\nDo not choose a model from accuracy alone. Check high-risk "
        "recall, because missing high-risk records is more concerning "
        "in a healthcare teaching example."
    )

    best_name = comparison.sort_values(
        ["F1 (macro)", "Accuracy"],
        ascending=False,
    )["Model"].iloc[0]
    print(f"\nHighest macro F1 on this split: {best_name}")
    save_trained_model(trained[best_name], best_name)
    return trained, comparison, best_name


def save_trained_model(model, model_name: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    payload = {
        "model": model,
        "model_name": model_name,
        "features": FEATURE_COLUMNS,
        "classes": CLASS_ORDER,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved trained model ({model_name}) to: {MODEL_PATH}")


def load_saved_model():
    if not MODEL_PATH.exists():
        return None
    payload = joblib.load(MODEL_PATH)
    return payload


def get_or_create_model():
    """Load a saved model, or train Random Forest if the GUI is opened first."""
    payload = load_saved_model()
    if payload is not None:
        return payload

    print("No saved model found. Training Random Forest from the local dataset...")
    df_raw = load_dataset()
    df = preprocess_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    save_trained_model(model, "Random Forest")
    return load_saved_model()


def features_to_frame(values: dict) -> pd.DataFrame:
    row = {column: [values[column]] for column in FEATURE_COLUMNS}
    return pd.DataFrame(row, columns=FEATURE_COLUMNS)


_cached_perf_details = None


def get_model_performance_details() -> dict:
    """Return performance metrics, confusion matrices, and figure paths for all models."""
    global _cached_perf_details
    if _cached_perf_details is not None:
        return _cached_perf_details

    df_raw = load_dataset()
    df = preprocess_dataset(df_raw)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models = build_models()

    details = {
        "models": {},
        "summary": [],
        "class_order": CLASS_ORDER,
        "test_counts": {k: int(v) for k, v in y_test.value_counts().reindex(CLASS_ORDER).items()},
        "test_total": len(y_test),
        "train_total": len(X_train),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        rep = classification_report(
            y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
        )

        safe_name = name.lower().replace(" ", "_")
        fig_path = FIGURES_DIR / f"06_confusion_{safe_name}.png"

        details["models"][name] = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "confusion_matrix": cm.tolist(),
            "report": rep,
            "figure_path": str(fig_path) if fig_path.exists() else None,
        }

        details["summary"].append({
            "Model": name,
            "Accuracy": f"{acc:.2%}",
            "Precision (macro)": f"{prec:.2%}",
            "Recall (macro)": f"{rec:.2%}",
            "F1 (macro)": f"{f1:.2%}",
            "High-risk Recall": f"{rep['high risk']['recall']:.2%}",
        })

    _cached_perf_details = details
    return details


def predict_risk(model, values: dict) -> dict:
    """Predict a dataset-defined RiskLevel from six numeric inputs."""
    frame = features_to_frame(values)
    predicted = str(model.predict(frame)[0])
    result = {"predicted_label": predicted, "probabilities": None}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        class_names = list(model.classes_)
        result["probabilities"] = {
            str(name): float(prob) for name, prob in zip(class_names, probs)
        }
    return result


def sample_predictions(model, model_name: str, X_test, y_test) -> None:
    print_section("Stage 8: Sample predictions")
    print(
        f"Using the {model_name} model. These are dataset-label predictions, "
        "not medical assessments."
    )

    sample = X_test.head(5).copy()
    true_labels = y_test.loc[sample.index]
    predicted = model.predict(sample)

    display = sample.copy()
    display["TrueRiskLevel"] = true_labels.values
    display["PredictedRiskLevel"] = predicted
    print("\nFive test-set examples:")
    print(display.to_string(index=False))

    # A single explicit feature vector taken from the first test row.
    first = sample.iloc[[0]]
    pred_one = model.predict(first)[0]
    print("\nSingle prediction from six input values:")
    for column in FEATURE_COLUMNS:
        print(f"  {column} = {first.iloc[0][column]}")
    print(f"  Predicted RiskLevel = {pred_one}")
    print(f"  Actual dataset label = {true_labels.iloc[0]}")


def interpret_results(df: pd.DataFrame, comparison: pd.DataFrame, best_name: str) -> None:
    print_section("Stage 9: Interpretation notes")
    print(f"Records used: {len(df)}")
    print(f"Features used: {FEATURE_COLUMNS}")
    print(f"Highest macro F1 model on this run: {best_name}")
    print("\nPrinted comparison table:")
    print(comparison.round(4).to_string(index=False))
    print("\nLimitations to mention in the report and viva:")
    print("- Labels come from the UCI dataset; they are not independently validated here.")
    print("- Six features cannot represent full maternal clinical assessment.")
    print("- Duplicate rows and unusual values (for example a very low heart rate) affect quality.")
    print("- High test accuracy does not mean the model is clinically safe.")
    print("- This project is educational only.")


def main() -> None:
    print("MHRLC: Maternal Health Risk Level Classification")
    print("Course: HDSSBL701 | Educational use only")

    df_raw = load_dataset()
    inspect_dataset(df_raw)
    df = preprocess_dataset(df_raw)
    run_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)
    trained, comparison, best_name = train_and_compare(
        X_train, X_test, y_train, y_test
    )
    sample_predictions(trained[best_name], best_name, X_test, y_test)
    interpret_results(df, comparison, best_name)

    print_section("Finished")
    print("Run completed. Figures are in the figures folder.")
    print("Re-run analysis with: python main.py")
    print("Open the prediction window with: python gui.py")


if __name__ == "__main__":
    main()
