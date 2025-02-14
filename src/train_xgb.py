#!/usr/bin/env python
import argparse
import os
import datetime
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import xgboost as xgb
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score, confusion_matrix,
                             recall_score, precision_score)

def get_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test

def train(model_name: str, max_depth: int, learning_rate: float, n_estimators: int,
          subsample: float, colsample_bytree: float, scale_pos_weight: float,
          gamma: float, min_child_weight: int, reg_alpha: float, reg_lambda: float,
          eval_metric: str, train_path: str, test_path: str, output_dir: str = "models"):

    os.makedirs(output_dir, exist_ok=True)
    df_train, df_test = get_data(train_path, test_path)
    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test  = df_test.drop(columns=['is_click'])
    y_test  = df_test['is_click']

    # Compute scale_pos_weight if not provided
    if scale_pos_weight is None:
        counts = y_train.value_counts()
        if 0 in counts and 1 in counts:
            scale_pos_weight = counts[0] / counts[1]
        else:
            scale_pos_weight = 1.0

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        eval_metric=eval_metric,
        objective='binary:logistic',
        booster='gbtree',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    bal_acc     = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary   = f1_score(y_test, y_pred, average='binary')
    prec        = precision_score(y_test, y_pred)
    rec         = recall_score(y_test, y_pred)
    kappa       = cohen_kappa_score(y_test, y_pred)
    cm          = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Balanced Accuracy: {bal_acc:.4f}")
    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")
    print(f"{model_name} - Precision: {prec:.4f}")
    print(f"{model_name} - Recall: {rec:.4f}")
    print(f"{model_name} - Cohen Kappa: {kappa:.4f}")

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    cm_fig_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()
    cm_table = wandb.Table(data=cm.tolist(), columns=["Predicted 0", "Predicted 1"])

    # Threshold metrics plot (no local reimport)
    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    recalls, precisions, bal_accs, f1s = [], [], [], []
    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        recalls.append(recall_score(y_test, y_pred_thresh))
        precisions.append(precision_score(y_test, y_pred_thresh))
        f1s.append(f1_score(y_test, y_pred_thresh))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred_thresh))
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, bal_accs, label='Balanced Accuracy')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Threshold')
    plt.legend()
    thresh_fig_path = os.path.join(output_dir, f"{model_name}_threshold_metrics.png")
    plt.savefig(thresh_fig_path)
    plt.close()
    thresh_img = wandb.Image(thresh_fig_path, caption="Metrics vs Threshold")

    wandb.log({
        "balanced_accuracy": bal_acc,
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "precision": prec,
        "recall": rec,
        "cohen_kappa": kappa,
        "confusion_matrix": cm_table,
        "confusion_matrix_image": wandb.Image(cm_fig_path, caption="Confusion Matrix"),
        "threshold_metrics_plot": thresh_img,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "eval_metric": eval_metric
    })

    incumbent_path = os.path.join(output_dir, f"{model_name}.joblib")
    incumbent_meta_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    replace_flag = False
    if os.path.exists(incumbent_path) and os.path.exists(incumbent_meta_path):
        incumbent_metadata = joblib.load(incumbent_meta_path)
        incumbent_bal_acc = incumbent_metadata.get("balanced_accuracy", 0)
        print(f"Incumbent {model_name} has Balanced Accuracy: {incumbent_bal_acc:.4f}")
        if bal_acc > incumbent_bal_acc:
            print("New XGBoost model is better. Replacing incumbent.")
            replace_flag = True
        else:
            print("New XGBoost model is not better. Saving candidate separately.")
    else:
        print("No incumbent XGBoost model found. Saving new model as incumbent.")
        replace_flag = True

    if replace_flag:
        joblib.dump(model, incumbent_path)
        metadata = {
            "balanced_accuracy": bal_acc,
            "f1_weighted": f1_weighted,
            "f1_binary": f1_binary,
            "precision": prec,
            "recall": rec,
            "cohen_kappa": kappa,
            "parameters": {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "scale_pos_weight": scale_pos_weight,
                "gamma": gamma,
                "min_child_weight": min_child_weight,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "eval_metric": eval_metric
            }
        }
        joblib.dump(metadata, incumbent_meta_path)
        print(f"Model saved as incumbent to {incumbent_path}")
        saved_path = incumbent_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = os.path.join(output_dir, f"{model_name}_{timestamp}.joblib")
        joblib.dump(model, candidate_path)
        print(f"Candidate model saved to {candidate_path}")
        saved_path = candidate_path

    return saved_path, bal_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Trainer")
    parser.add_argument("-m", "--model-name", default="XGBoost")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help="If not provided, will be computed as (# negatives)/(# positives)")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--min-child-weight", type=int, default=1)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--eval-metric", type=str, default="logloss")
    parser.add_argument("-o", "--output-dir", default="/home/matangold/ydata/models")
    parser.add_argument("-t", "--train-path", default="/home/matangold/ydata/data/train.csv")
    parser.add_argument("-p", "--test-path", default="/home/matangold/ydata/data/test.csv")
    args = parser.parse_args()

    wandb.init(project="model-training", name=args.model_name, config=vars(args))
    train(args.model_name, args.max_depth, args.learning_rate, args.n_estimators, args.subsample,
          args.colsample_bytree, args.scale_pos_weight, args.gamma, args.min_child_weight,
          args.reg_alpha, args.reg_lambda, args.eval_metric, args.train_path, args.test_path, args.output_dir)
