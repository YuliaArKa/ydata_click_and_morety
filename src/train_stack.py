#!/usr/bin/env python
import argparse
import os
import glob
import joblib
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score, confusion_matrix)

def get_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test

def load_base_models(model_dir, model_keys):
    """
    Loads base models from model_dir using filename patterns.
    model_keys is a list of keys (e.g., ['lightgbm', 'random_forest']) to load.
    """
    all_patterns = {
        'lightgbm': os.path.join(model_dir, 'LightGBM.joblib'),
        'random_forest': os.path.join(model_dir, 'RandomForest.joblib'),
        'xgboost': os.path.join(model_dir, 'XGBoost.joblib'),
        'logistic_regression': os.path.join(model_dir, 'LogisticRegression.joblib'),
        'naive_bayes': os.path.join(model_dir, 'NaiveBayes.joblib'),
        'knn': os.path.join(model_dir, 'Knn.joblib'),
        'dt': os.path.join(model_dir, 'DecisionTree.joblib'),
        'ada': os.path.join(model_dir, 'AdaBoost.joblib')
    }
    base_models = {}
    for key in model_keys:
        pattern = all_patterns.get(key)
        if not pattern:
            print(f"Warning: Unknown base model key '{key}'. Skipping.")
            continue
        files = glob.glob(pattern)
        if not files:
            print(f"Error: Could not find a base model matching: {pattern}")
            exit(1)
        base_models[key] = joblib.load(files[0])
        print(f"Loaded base model '{key}' from {files[0]}")
    return base_models

def create_meta_features(X, base_models):
    """Creates meta features by obtaining probability predictions from each base model."""
    meta_features = pd.DataFrame()
    for key, model in base_models.items():
        # Use probability of the positive class.
        meta_features[f'{key}_pred'] = model.predict_proba(X)[:, 1]
    return meta_features

def train_stacking(model_name: str, base_model_dir: str, base_models_list: list,
                   train_path: str, test_path: str, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    df_train, df_test = get_data(train_path, test_path)
    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test  = df_test.drop(columns=['is_click'])
    y_test  = df_test['is_click']

    # Load only the base models specified via the command line.
    base_models = load_base_models(base_model_dir, base_models_list)
    meta_X_train = create_meta_features(X_train, base_models)
    meta_X_test  = create_meta_features(X_test, base_models)

    # Train a Logistic Regression meta model.
    meta_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=500)
    meta_model.fit(meta_X_train, y_train)
    y_pred = meta_model.predict(meta_X_test)

    # Compute evaluation metrics.
    bal_acc    = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary   = f1_score(y_test, y_pred, average='binary')
    prec       = precision_score(y_test, y_pred)
    rec        = recall_score(y_test, y_pred)
    kappa      = cohen_kappa_score(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Balanced Accuracy: {bal_acc:.4f}")
    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")
    print(f"{model_name} - Precision: {prec:.4f}")
    print(f"{model_name} - Recall: {rec:.4f}")
    print(f"{model_name} - Cohen Kappa: {kappa:.4f}")

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    cm_fig_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()
    cm_table = wandb.Table(data=cm.tolist(), columns=["Predicted 0", "Predicted 1"])

    wandb.log({
        "balanced_accuracy": bal_acc,
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "precision": prec,
        "recall": rec,
        "cohen_kappa": kappa,
        "confusion_matrix": cm_table,
        "confusion_matrix_image": wandb.Image(cm_fig_path, caption="Confusion Matrix"),
        "model_name": model_name,
        "base_models_used": base_models_list
    })

    # Incumbent logic (save model if it improves)
    incumbent_path = os.path.join(output_dir, f"{model_name}.joblib")
    incumbent_meta_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    replace_flag = False
    if os.path.exists(incumbent_path) and os.path.exists(incumbent_meta_path):
        incumbent_metadata = joblib.load(incumbent_meta_path)
        incumbent_bal_acc = incumbent_metadata.get("balanced_accuracy", 0)
        print(f"Incumbent {model_name} has Balanced Accuracy: {incumbent_bal_acc:.4f}")
        if bal_acc > incumbent_bal_acc:
            print("New stacking model is better. Replacing incumbent.")
            replace_flag = True
        else:
            print("New stacking model is not better. Saving candidate separately.")
    else:
        print("No incumbent stacking model found. Saving new model as incumbent.")
        replace_flag = True

    if replace_flag:
        joblib.dump(meta_model, incumbent_path)
        metadata = {
            "balanced_accuracy": bal_acc,
            "f1_weighted": f1_weighted,
            "f1_binary": f1_binary,
            "precision": prec,
            "recall": rec,
            "cohen_kappa": kappa,
            "parameters": f"Base models: {base_models_list}; Meta model: LogisticRegression"
        }
        joblib.dump(metadata, incumbent_meta_path)
        print(f"Stacking model saved as incumbent to {incumbent_path}")
        saved_path = incumbent_path
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = os.path.join(output_dir, f"{model_name}_{timestamp}.joblib")
        joblib.dump(meta_model, candidate_path)
        print(f"Candidate stacking model saved to {candidate_path}")
        saved_path = candidate_path

    return saved_path, bal_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking Meta Model Trainer")
    parser.add_argument("-m", "--model-name", default="StackingLR")
    parser.add_argument("-b", "--base-model-dir", default="/home/matangold/ydata/models",
                        help="Directory containing base models")
    parser.add_argument("--base-models", type=str, default="lightgbm,random_forest,xgboost,logistic_regression,naive_bayes,knn,dt",
                        help="Comma-separated list of base model keys to use (e.g., lightgbm,random_forest,xgboost)")
    parser.add_argument("-o", "--output-dir", default="/home/matangold/ydata/models")
    parser.add_argument("-t", "--train-path", default="/home/matangold/ydata/data/train.csv")
    parser.add_argument("-p", "--test-path", default="/home/matangold/ydata/data/test.csv")
    args = parser.parse_args()

    base_models_list = [s.strip() for s in args.base_models.split(",") if s.strip()]

    wandb.init(project="model-training", name=args.model_name, config=vars(args))
    train_stacking(args.model_name, args.base_model_dir, base_models_list, args.train_path, args.test_path, args.output_dir)
