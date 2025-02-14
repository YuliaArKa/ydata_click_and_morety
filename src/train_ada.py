# train_adaboost.py
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import joblib
import os
import datetime
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score, confusion_matrix)


def get_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Loads the training and test datasets from the specified paths."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def train(model_name: str, n_estimators: int, learning_rate: float, output_dir: str = "models",
          train_path: str = "data/train.csv", test_path: str = "data/test.csv"):
    os.makedirs(output_dir, exist_ok=True)
    df_train, df_test = get_data(train_path, test_path)

    X_train = df_train.drop(columns=['is_click'])
    y_train = df_train['is_click']
    X_test = df_test.drop(columns=['is_click'])
    y_test = df_test['is_click']

    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_binary = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Balanced Accuracy: {bal_acc:.4f}")
    print(f"{model_name} - Weighted F1 Score: {f1_weighted:.4f}")
    print(f"{model_name} - Binary F1 Score: {f1_binary:.4f}")
    print(f"{model_name} - Precision: {precision:.4f}")
    print(f"{model_name} - Recall: {recall:.4f}")
    print(f"{model_name} - Cohen Kappa: {kappa:.4f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    wandb.log({
        "balanced_accuracy": bal_acc,
        "f1_weighted": f1_weighted,
        "f1_binary": f1_binary,
        "precision": precision,
        "recall": recall,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "model_name": model_name,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    })

    incumbent_path = os.path.join(output_dir, f"{model_name}.joblib")
    incumbent_meta_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    replace_flag = False
    if os.path.exists(incumbent_path) and os.path.exists(incumbent_meta_path):
        incumbent_metadata = joblib.load(incumbent_meta_path)
        incumbent_bal_acc = incumbent_metadata.get("balanced_accuracy", 0)
        print(f"Incumbent {model_name} has Balanced Accuracy: {incumbent_bal_acc:.4f}")
        if bal_acc > incumbent_bal_acc:
            print("New AdaBoost model is better. Replacing incumbent.")
            replace_flag = True
        else:
            print("New AdaBoost model is not better. Saving candidate separately.")
    else:
        print("No incumbent AdaBoost model found. Saving new model as incumbent.")
        replace_flag = True

    if replace_flag:
        joblib.dump(model, incumbent_path)
        metadata = {
            "balanced_accuracy": bal_acc,
            "f1_weighted": f1_weighted,
            "f1_binary": f1_binary,
            "precision": precision,
            "recall": recall,
            "cohen_kappa": kappa,
            "parameters": {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate
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
    parser = argparse.ArgumentParser(description="AdaBoost Trainer")
    parser.add_argument("-m", "--model-name", default="AdaBoost", help="Name of the model")
    parser.add_argument("-n", "--n-estimators", type=int, default=50, help="Number of estimators for AdaBoost")
    parser.add_argument("-r", "--learning-rate", type=float, default=1.0, help="Learning rate for AdaBoost")
    parser.add_argument("-o", "--output-dir", default="models", help="Directory to save the trained model")
    parser.add_argument("-t", "--train-path", type=str, default="data/train.csv", help="Path to training data")
    parser.add_argument("-T", "--test-path", type=str, default="data/test.csv", help="Path to testing data")
    args = parser.parse_args()

    wandb.init(project="model-training", name=args.model_name, config=vars(args))
    train(args.model_name, args.n_estimators, args.learning_rate, args.output_dir, args.train_path, args.test_path)
