import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import optuna
import shap
import joblib
import os
import pandas as pd
import numpy as np


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


def objective(trial, X_train, y_train, skf):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'verbose': -1
    }

    auc_scores = []
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_fold_train, y_fold_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = LGBMClassifier(**param)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_valid, y_fold_valid)], eval_metric='auc')

        y_pred_proba = model.predict_proba(X_fold_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_fold_valid, y_pred_proba))

    return np.mean(auc_scores)


def optimize_and_train(pro_data, pro_labels, output_dir, test_size=0.3, n_trials=50):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for disease in pro_labels.columns:
        print(f"Processing {disease}...")
        y = pro_labels[disease]
        X = pro_data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, skf), n_trials=n_trials)

        print(f"Best parameters for {disease}: {study.best_params}")
        print(f"Best AUC during validation: {study.best_value}")

        best_model = LGBMClassifier(**study.best_params)
        best_model.fit(X_train, y_train)

        model_path = os.path.join(output_dir, f"{disease}_best_lightgbm_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"模型保存至: {model_path}")

        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        lower, upper = calculate_auc_confidence_interval(y_test.values, y_test_proba)
        print(f"{disease} Test AUC: {test_auc} (95% CI: [{lower:.3f}, {upper:.3f}])")

        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)

        shap.summary_plot(shap_values, X_train, show=False)
        shap_plot_path = os.path.join(output_dir, f"{disease}_shap_summary.pdf")
        plt.savefig(shap_plot_path, format='pdf')
        plt.close()

        shap_values_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
        shap_values_df_path = os.path.join(output_dir, f"{disease}_shap_values.csv")
        shap_values_df.to_csv(shap_values_df_path, index=False)

        results.append({
            'Disease': disease,
            'Validation AUC': study.best_value,
            'Test AUC': test_auc,
            'AUC Lower 95% CI': lower,
            'AUC Upper 95% CI': upper,
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

    optimize_and_train(
        pro_data,
        pro_labels,
        output_dir="/work/sph-xutx/codes/immune/sle_models/lightgbm/result2",
        test_size=0.3,
        n_trials=50
    )
