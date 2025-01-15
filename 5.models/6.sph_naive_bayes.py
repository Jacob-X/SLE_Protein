import os
import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


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


def train_and_evaluate_naive_bayes(pro_data, pro_labels, output_dir, disease):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"Processing {disease}...")

    y = pro_labels[disease]
    X = pro_data

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model = GaussianNB()

    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, y_pred_proba)

    lower, upper = calculate_auc_confidence_interval(y.values, y_pred_proba)
    print(f"{disease} AUC: {auc}, 95% CI: ({lower}, {upper})")

    model.fit(X, y)

    model_path = os.path.join(output_dir, f"{disease}_best_nb_model.pkl")
    joblib.dump(model, model_path)
    print(f"模型保存至: {model_path}")

    results.append({
        'Disease': disease,
        'AUC': auc,
        'AUC 95% CI Lower': lower,
        'AUC 95% CI Upper': upper,
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
    output_dir = "/work/sph-xutx/codes/immune/sle_models/naive_bayes/result"

    train_and_evaluate_naive_bayes(pro_data, pro_labels, output_dir, disease)

#
# #!/bin/bash
# set -e
# module load python/anaconda3/5.2.0
# export PATH=/work/sph-xutx/.conda/envs/lgb/bin:$PATH
# export CONDA_PREFIX=/work/sph-xutx/.conda/envs/lgb
# python /work/sph-xutx/codes/immune/sle_models/naive_bayes/6.sph_naive_bayes.py

# bsub -q short -n 40 -J sph_naive_bayes -o sph_naive_bayes.LOG -e sph_naive_bayes.ERR < sph_naive_bayes.sh