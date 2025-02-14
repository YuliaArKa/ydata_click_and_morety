import argparse
import pandas as pd
import joblib
import os
import glob
import sys
import subprocess
from importlib.util import spec_from_file_location, module_from_spec


# ----------------------------
# Stacking Predictor Wrapper
# ----------------------------
class StackingPredictor:
    def __init__(self, base_models, meta_model, variant='with_features'):
        self.base_models = base_models
        self.meta_model = meta_model
        self.variant = variant

    def predict(self, X):
        preds = {key: model.predict_proba(X)[:, 1] for key, model in self.base_models.items()}
        df_meta = pd.DataFrame(preds)

        meta_features = pd.concat([X.reset_index(drop=True), df_meta.reset_index(drop=True)],
                                  axis=1) if self.variant == 'with_features' else df_meta
        return self.meta_model.predict(meta_features.values)


# ----------------------------
# Data Loading Function
# ----------------------------
def load_data(file_path):
    return pd.read_csv(file_path)


# ----------------------------
# Base Models Loader
# ----------------------------
def load_base_models(model_dir):
    model_filenames = {
        'lightgbm': 'LightGBM.joblib',
        'random_forest': 'RandomForest.joblib',
        'xgboost': 'XGBoost.joblib',
        'logistic_regression': 'LogisticRegression.joblib',
        'knn': 'Knn.joblib',
        'adaboost': 'AdaBoost.joblib',
        'naive_bayes': 'NaiveBayes.joblib',
        'decision_tree': 'DecisionTree.joblib'
    }

    base_models = {}
    for key, filename in model_filenames.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            base_models[key] = joblib.load(model_path)
            print(f"Loaded {key} model from {model_path}")
        else:
            print(f"Warning: {model_path} not found, skipping {key}.")
    return base_models


# ----------------------------
# Import Preprocess Module Dynamically
# ----------------------------
def import_preprocess_module(script_path):
    spec = spec_from_file_location("preprocess", script_path)
    preprocess_module = module_from_spec(spec)
    spec.loader.exec_module(preprocess_module)
    return preprocess_module


# ----------------------------
# Prediction Execution
# ----------------------------
def predict_and_save(model_name, meta_model_path, output_csv_path, variant, data_path, model_dir, preprocess=False,
                     train_path=None, raw_data=None):
    if preprocess:
        # Import preprocess module
        preprocess_script = os.path.join(os.path.dirname(__file__), "preprocess.py")
        preprocess_module = import_preprocess_module(preprocess_script)

        # Get preprocessed data directly
        X_test_dummies, _ = preprocess_module.preprocess_data("predict", train_path, raw_data)
        X_test = X_test_dummies
        X_test
    else:
        X_test = load_data(data_path)
        X_test = X_test.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col, axis=0)

    if model_name == 'stacking':
        print(f"Loading meta model from {meta_model_path}")
        meta_model = joblib.load(meta_model_path)
        base_models = load_base_models(model_dir)
        predictor = StackingPredictor(base_models, meta_model, variant)
        y_pred = predictor.predict(X_test)
    else:
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"Error: Model {model_name} not found at {model_path}")
            sys.exit(1)
        print(f"Using base model {model_name} from {model_path}")
        model = joblib.load(model_path)
        y_pred = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame(y_pred)
    submission.to_csv(output_csv_path, index=False, header=False)
    print(f"Predictions saved to {output_csv_path}")


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using stacking ensemble or base models")
    parser.add_argument("-m", "--meta-model-path", required=False,
                        help="Path to the meta model file (used for stacking)")
    parser.add_argument("-o", "--output-csv-path", required=True, help="Path to save predictions CSV")
    parser.add_argument("-v", "--variant", choices=['with_features', 'without_features'], default='with_features',
                        help="Stacking variant")
    parser.add_argument("-d", "--data-path", required=True, help="Path to the input data CSV")
    parser.add_argument("-b", "--base-model", required=True,
                        help="Specify the base model name or 'stacking' for ensemble")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Run preprocessing before prediction")
    parser.add_argument("-r", "--raw-data", required=False,
                        help="Path to raw data (needed if preprocessing is enabled)")
    parser.add_argument("-model-dir", required=False, default="models", help="Directory containing trained models")
    parser.add_argument("--train-path", required=False,
                        help="Path to training data (needed if preprocessing is enabled)")

    args = parser.parse_args()

    if args.preprocess and (not args.raw_data or not args.train_path):
        print("Error: When using -p/--preprocess, both --raw-data and --train-path are required")
        sys.exit(1)

    predict_and_save(
        args.base_model,
        args.meta_model_path,
        args.output_csv_path,
        args.variant,
        args.data_path,
        args.model_dir,
        preprocess=args.preprocess,
        train_path=args.train_path,
        raw_data=args.raw_data
    )