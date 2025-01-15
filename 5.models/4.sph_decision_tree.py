import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import os
import optuna
import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def calculate_auc_confidence_interval(y_true, y_pred_proba, confidence_level=0.95):
    auc_scores = []
    n_bootstrap = 100
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        auc_scores.append(roc_auc_score(y_true[indices], y_pred_proba[indices]))

    lower = np.percentile(auc_scores, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(auc_scores, (1 + confidence_level) / 2 * 100)
    return lower, upper


def objective(trial, X, y, skf):
    param = {
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 3407
    }

    auc_scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = DecisionTreeClassifier(**param)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, y_pred_proba))

    return np.mean(auc_scores)


def optimize_decision_tree_with_optuna(pro_data, pro_labels, output_dir, disease, n_trials=50):

    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"Processing {disease}...")

    y = pro_labels[disease]
    X = pro_data
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, skf), n_trials=n_trials)

    print(f"Best parameters for {disease}: {study.best_params}")
    print(f"Best AUC for {disease}: {study.best_value}")

    best_model = DecisionTreeClassifier(**study.best_params)
    best_model.fit(X, y)


    model_path = os.path.join(output_dir, f"{disease}_best_dt_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"模型保存至: {model_path}")

    y_pred_proba = cross_val_predict(best_model, X, y, cv=skf, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, y_pred_proba)

    lower, upper = calculate_auc_confidence_interval(y.values, y_pred_proba)
    print(f"{disease} AUC: {auc}, 95% CI: ({lower}, {upper})")

    results.append({
        'Disease': disease,
        'AUC': auc,
        'AUC 95% CI Lower': lower,
        'AUC 95% CI Upper': upper,
        'Best Parameters': study.best_params,
        'Model Path': model_path
    })

    results_df = pd.DataFrame(results)
    results_df_path = os.path.join(output_dir, "model_performance.csv")
    results_df.to_csv(results_df_path, index=False)

    print(f"所有结果已保存至 {output_dir}.")


if __name__ == '__main__':

    pro = pd.read_csv("/work/sph-xutx/codes/immune/immune_lgb_feature_selection/immune_pro.csv")
    pro = pro.fillna(pro.median(numeric_only=True))

    pro_data = pro[['trim21', 'il15', 'scarb2', 'lgals9', 'pdcd1', 'sod2', 'mepe', 'lag3', 'cxcl16', 'pomc', 'bst2']]
    pro_labels = pro[["SLE"]]

    disease = "SLE"
    output_dir = "/work/sph-xutx/codes/immune/sle_models/decision_tree/result"

    optimize_decision_tree_with_optuna(pro_data, pro_labels, output_dir, disease, n_trials=50)

#
# #!/bin/bash
# set -e
# module load python/anaconda3/5.2.0
# export PATH=/work/sph-xutx/.conda/envs/lgb/bin:$PATH
# export CONDA_PREFIX=/work/sph-xutx/.conda/envs/lgb
# python /work/sph-xutx/codes/immune/sle_models/decision_tree/4.sph_decision_tree.py

# bsub -q short -n 40 -J sph_decision_tree -o sph_decision_tree.LOG -e sph_decision_tree.ERR < sph_decision_tree.sh