from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import pandas as pd
import wandb

wandb.init(
    project="CTR_Prediction_YData_Phase2",
    name="model_training_baseline",
    config={
        "test_size": 0.2,
        "random_state": 7
    }
)


def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=10, class_weight="balanced"),
        "NaiveBayes": GaussianNB(),
        "LightGBM": LGBMClassifier(boosting_type='gbdt', is_unbalance=True, learning_rate=0.05, max_depth=10, n_estimators=100)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        wandb.log({f"{name}_trained": True})
        print(f"Trained {name}")
    
    return models

if __name__ == "__main__":
    df_train = pd.read_csv("data/train_final.csv")
    y_train = df_train.pop("is_click")
    X_train, X_val, y_train, y_val = train_test_split(df_train, y_train, test_size=0.2, random_state=42)
    
    models = train_models(X_train, y_train)
    print("Model training completed!")
