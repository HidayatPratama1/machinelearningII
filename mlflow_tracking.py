# =============================================================
# mlflow_tracking.py -- Tahap 9: Tracking Eksperimen MLflow
# Dataset : IoT Vulnerability (Preprocessed Balanced)
# Metode  : Filter (SelectKBest, k=38) + RandomForestClassifier
# =============================================================

# -- Install otomatis jika belum ada --------------------------
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

required = ["mlflow", "scikit-learn", "pandas", "numpy", "joblib", "imbalanced-learn"]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# -- Import ---------------------------------------------------
import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)

# =============================================================
# KONFIGURASI -- sesuaikan path jika diperlukan
# =============================================================
DATASET_PATH = "F:\\Teknik informarika\\Semester 6 03PT6\\Machine Learning II\\MID\\MID\\Preprocessed_Balanced_dataset.csv"
RANDOM_STATE = 42

# -- Parameter TERBAIK dari GridSearchCV (Tahap 7) ------------
BEST_PARAMS = {
    "feature_selection__k"     : 38,
    "model__n_estimators"      : 100,
    "model__max_depth"         : None,   # None = unlimited
    "model__min_samples_split" : 2,
}

# Metrik final dari test set (Tahap 8 notebook)
FINAL_METRICS = {
    "test_accuracy"    : 1.0000,
    "test_f1_macro"    : 1.0000,
    "test_precision"   : 1.0000,
    "test_recall"      : 1.0000,
    "cv_f1_macro_mean" : 1.0000,
    "cv_f1_macro_std"  : 0.0000,
}

# =============================================================
# SETUP MLflow
# =============================================================
EXPERIMENT_NAME = "IoT_Vulnerability_FeatureSelection"
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print(" MLflow Tracking -- IoT Vulnerability Detection")
print("=" * 60)
print(f"Experiment  : {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# =============================================================
# [1/4] LOAD & PREPROCESS DATA
# =============================================================
print(f"\n[1/4] Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"      Shape: {df.shape}")

target_col = "Label"
le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df[target_col])

# Encode kolom kategorik (Attack_sub_category)
cat_cols = (df.drop(columns=[target_col, "target_encoded"])
              .select_dtypes(include=["object"]).columns)
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=[target_col, "target_encoded"])
y = df["target_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

# =============================================================
# [2/4] BANGUN & LATIH PIPELINE TERBAIK (parameter statis)
# =============================================================
print("\n[2/4] Membangun pipeline terbaik (best params statis)...")

best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=BEST_PARAMS["feature_selection__k"]
    )),
    ("model", RandomForestClassifier(
        n_estimators      = BEST_PARAMS["model__n_estimators"],
        max_depth         = BEST_PARAMS["model__max_depth"],
        min_samples_split = BEST_PARAMS["model__min_samples_split"],
        random_state      = RANDOM_STATE,
        n_jobs            = -1
    ))
])

print("      Training pipeline...")
best_pipeline.fit(X_train, y_train)
print("      Training selesai.")

# Verifikasi metrik pada test set
y_pred = best_pipeline.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="macro")
prec = precision_score(y_test, y_pred, average="macro")
rec  = recall_score(y_test, y_pred, average="macro")

print(f"\n      Verifikasi metrik test set:")
print(f"        Accuracy  : {acc:.4f}")
print(f"        F1 Macro  : {f1:.4f}")
print(f"        Precision : {prec:.4f}")
print(f"        Recall    : {rec:.4f}")

# =============================================================
# [3/4] MLflow RUN
# =============================================================
print("\n[3/4] Memulai MLflow run...")

with mlflow.start_run(run_name="best_pipeline_filter_selectkbest") as run:

    # -- Tags -------------------------------------------------
    mlflow.set_tags({
        "dataset"          : "IoT Vulnerability Preprocessed Balanced",
        "feature_selection": "Filter - SelectKBest (f_classif)",
        "model"            : "RandomForestClassifier",
        "task"             : "Binary Classification",
        "author"           : "ML II Group",
        "phase"            : "Best Model - Final",
        "n_classes"        : "2",
    })

    # -- Log Parameters ---------------------------------------
    mlflow.log_params({
        "fs_method"           : "SelectKBest",
        "fs_score_func"       : "f_classif",
        "fs_k"                : BEST_PARAMS["feature_selection__k"],
        "model_type"          : "RandomForestClassifier",
        "n_estimators"        : BEST_PARAMS["model__n_estimators"],
        "max_depth"           : str(BEST_PARAMS["model__max_depth"]),
        "min_samples_split"   : BEST_PARAMS["model__min_samples_split"],
        "random_state"        : RANDOM_STATE,
        "n_features_original" : X.shape[1],
        "n_features_selected" : BEST_PARAMS["feature_selection__k"],
        "train_size"          : X_train.shape[0],
        "test_size"           : X_test.shape[0],
        "test_ratio"          : 0.2,
        "cv_folds"            : 5,
        "cv_strategy"         : "StratifiedKFold",
    })

    # -- Log Metrics ------------------------------------------
    mlflow.log_metrics({
        # Dari Tahap 8 notebook (ground truth)
        "test_accuracy"        : FINAL_METRICS["test_accuracy"],
        "test_f1_macro"        : FINAL_METRICS["test_f1_macro"],
        "test_precision_macro" : FINAL_METRICS["test_precision"],
        "test_recall_macro"    : FINAL_METRICS["test_recall"],
        # Dari Tahap 6 notebook (CV)
        "cv_f1_macro_mean"     : FINAL_METRICS["cv_f1_macro_mean"],
        "cv_f1_macro_std"      : FINAL_METRICS["cv_f1_macro_std"],
        # Re-computed dari re-train script ini (verifikasi)
        "recomputed_accuracy"  : acc,
        "recomputed_f1_macro"  : f1,
        "recomputed_precision" : prec,
        "recomputed_recall"    : rec,
    })

    # -- Artifact 1: Classification Report --------------------
    report_str  = classification_report(
        y_test, y_pred,
        target_names=[f"Class {c}" for c in le.classes_]
    )
    report_path = "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("IoT Vulnerability - Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset : Preprocessed Balanced (1,048,575 samples)\n")
        f.write(f"Features: {X.shape[1]} original -> {BEST_PARAMS['feature_selection__k']} selected\n")
        f.write(f"Method  : SelectKBest (f_classif, k={BEST_PARAMS['feature_selection__k']})\n")
        f.write(f"Model   : RandomForest (n_estimators={BEST_PARAMS['model__n_estimators']})\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_str)
    mlflow.log_artifact(report_path)
    print(f"      Artifact logged: {report_path}")

    # -- Artifact 2: Feature Importance CSV -------------------
    fs_step        = best_pipeline.named_steps["feature_selection"]
    selected_mask  = fs_step.get_support()
    selected_feats = np.array(X.columns)[selected_mask]
    importances    = best_pipeline.named_steps["model"].feature_importances_

    feat_df = pd.DataFrame({
        "feature"    : selected_feats,
        "importance" : importances,
        "f_score"    : fs_step.scores_[selected_mask],
    }).sort_values("importance", ascending=False)

    feat_csv = "feature_importance.csv"
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8")
    mlflow.log_artifact(feat_csv)
    print(f"      Artifact logged: {feat_csv}")

    # -- Artifact 3: Best Params JSON -------------------------
    params_path = "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in BEST_PARAMS.items()}, f, indent=2)
    mlflow.log_artifact(params_path)
    print(f"      Artifact logged: {params_path}")

    # -- Log Model (sklearn flavor) ---------------------------
    mlflow.sklearn.log_model(
        sk_model              = best_pipeline,
        artifact_path         = "pipeline_model",
        registered_model_name = "IoT_Vulnerability_Pipeline",
    )
    print("      Model logged ke MLflow registry.")

    # -- Juga simpan .pkl sebagai artifact --------------------
    pkl_path = "pipeline_terbaik_mlflow.pkl"
    joblib.dump(best_pipeline, pkl_path)
    mlflow.log_artifact(pkl_path)
    print(f"      Artifact logged: {pkl_path}")

    run_id = run.info.run_id
    print(f"\n  MLflow run berhasil!")
    print(f"  Run ID   : {run_id}")
    print(f"  Artifact : {mlflow.get_artifact_uri()}")

# =============================================================
# [4/4] RINGKASAN
# =============================================================
print("\n" + "=" * 60)
print(" RINGKASAN MLflow RUN")
print("=" * 60)
print(f" Experiment   : {EXPERIMENT_NAME}")
print(f" Run ID       : {run_id}")
print(f" Method       : Filter - SelectKBest (k=38)")
print(f" n_estimators : {BEST_PARAMS['model__n_estimators']}")
print(f" max_depth    : {BEST_PARAMS['model__max_depth']}")
print(f" Accuracy     : {FINAL_METRICS['test_accuracy']:.4f}")
print(f" F1 Macro     : {FINAL_METRICS['test_f1_macro']:.4f}")
print("=" * 60)
print("\n>> Jalankan MLflow UI dengan perintah:")
print("   mlflow ui")
print("   Kemudian buka: http://127.0.0.1:5000")
print("=" * 60)
