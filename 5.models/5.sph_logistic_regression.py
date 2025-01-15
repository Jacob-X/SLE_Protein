import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import os
import optuna
import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


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
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
    param = {
        'penalty': penalty,
        'C': trial.suggest_float('C', 1e-4, 10, log=True),
        'solver': 'saga',
        'class_weight': 'balanced',
        'random_state': 3407,
        'max_iter': 1000,
    }

    if penalty == 'elasticnet':
        param['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    auc_scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LogisticRegression(**param)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, y_pred_proba))

    return np.mean(auc_scores)


def optimize_logistic_regression_with_optuna(pro_data, pro_labels, output_dir, disease, n_trials=50):

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

    best_model = LogisticRegression(**study.best_params)
    best_model.fit(X, y)

    model_path = os.path.join(output_dir, f"{disease}_best_lr_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"模型保存至: {model_path}")

    y_pred_proba = cross_val_predict(best_model, X, y, cv=skf, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, y_pred_proba)

    lower, upper = calculate_auc_confidence_interval(y.values, y_pred_proba)
    print(f"{disease} AUC: {auc}, 95% CI: ({lower}, {upper})")

    explainer = shap.Explainer(best_model, X)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    shap_plot_path = os.path.join(output_dir, f"{disease}_shap_summary.png")
    plt.savefig(shap_plot_path)
    plt.close()

    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_values_df_path = os.path.join(output_dir, f"{disease}_shap_values.csv")
    shap_values_df.to_csv(shap_values_df_path, index=False)

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
    output_dir = "/work/sph-xutx/codes/immune/sle_models/logistic_regression/result"

    optimize_logistic_regression_with_optuna(pro_data, pro_labels, output_dir, disease, n_trials=50)

#
# #!/bin/bash
# set -e
# module load python/anaconda3/5.2.0
# export PATH=/work/sph-xutx/.conda/envs/lgb/bin:$PATH
# export CONDA_PREFIX=/work/sph-xutx/.conda/envs/lgb
# python /work/sph-xutx/codes/immune/sle_models/logistic_regression/5.sph_logistic_regression.py

# bsub -q short -n 40 -J sph_logistic_regression -o sph_logistic_regression.LOG -e sph_logistic_regression.ERR < sph_logistic_regression.sh