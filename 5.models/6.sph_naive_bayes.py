import os
import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def train_and_evaluate_naive_bayes(pro_data, pro_labels, output_dir, disease, test_size=0.3):

    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"Processing {disease}...")

    y = pro_labels[disease]
    X = pro_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    model = GaussianNB()

    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    lower, upper = calculate_auc_confidence_interval(y_test.values, y_test_proba)
    print(f"{disease} Test AUC: {test_auc:.3f}, 95% CI: ({lower:.3f}, {upper:.3f})")

    model_path = os.path.join(output_dir, f"{disease}_best_nb_model.pkl")
    joblib.dump(model, model_path)
    print(f"模型保存至: {model_path}")

    results.append({
        'Disease': disease,
        'Test AUC': test_auc,
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
    output_dir = "/work/sph-xutx/codes/immune/sle_models/naive_bayes/result2"

    train_and_evaluate_naive_bayes(pro_data, pro_labels, output_dir, disease, test_size=0.3)
