import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

DEFAULT_PREDICTION_CSV = "predictions.csv"
ARCHIVED_EXPERIMENTS_DIR = "archived_experiments"
DATE_TIME_PATTERN = "%Y%m%d_%H%M%S"  # adjust as needed

def calculate_classification_report(df):
    """Calculate and format classification metrics"""
    y_true = df['is_click']
    y_pred = df['prediction']

    # Calculate metrics
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_binary = f1_score(y_true, y_pred, average='binary')

    # Create confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save plot
    timestamp = datetime.now().strftime(DATE_TIME_PATTERN)
    cm_path = os.path.join("results", f'confusion_matrix_{timestamp}.png')
    os.makedirs("results", exist_ok=True)
    plt.savefig(cm_path)
    plt.close()

    # Format results
    results = {
        'metrics': {
            'f1_weighted': float(f1_weighted),
            'f1_binary': float(f1_binary),
            'precision': float(report_dict['weighted avg']['precision']),
            'recall': float(report_dict['weighted avg']['recall']),
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'timestamp': timestamp
    }
    return results

# --- Archive task (if using a task runner like Invoke) ---
# This snippet shows how you might archive experiments.
# Adjust or remove as needed.
#
# from invoke import task
#
# @task
# def archive(c, name, base_folder=ARCHIVED_EXPERIMENTS_DIR):
#    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"
#    print(f"Archiving experiment: {name}")
#    if not os.path.exists(base_folder):
#        os.makedirs(base_folder)
#
#    exp_path = os.path.join(base_folder, name)
#    c.run(f"mkdir {exp_path}")
#
#    # Archive folders if they exist
#    for folder in ["data", "results", "models"]:
#        if os.path.exists(folder):
#            c.run(f"mv {folder} {exp_path}")
#            os.makedirs(folder, exist_ok=True)

def results(csv_path, per_dataset=False, output_dir="results"):
    """Process and save model prediction results"""
    os.makedirs(output_dir, exist_ok=True)
    predictions_df = pd.read_csv(csv_path)

    if per_dataset:
        datasets = predictions_df["dataset_name"].unique().tolist()
        results_dict = {}

        for dataset in datasets:
            df = predictions_df[predictions_df["dataset_name"] == dataset]
            results_dict[dataset] = calculate_classification_report(df)

            # Save individual dataset results
            dataset_file = os.path.join(output_dir, f'results_{dataset}_{results_dict[dataset]["timestamp"]}.json')
            with open(dataset_file, 'w') as f:
                json.dump(results_dict[dataset], f, indent=4)

            print(f"\nResults for dataset {dataset}:")
            print(f"F1 Weighted: {results_dict[dataset]['metrics']['f1_weighted']:.4f}")
            print(f"F1 Binary: {results_dict[dataset]['metrics']['f1_binary']:.4f}")
    else:
        results_dict = calculate_classification_report(predictions_df)

        # Save overall results
        results_file = os.path.join(output_dir, f'results_overall_{results_dict["timestamp"]}.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print("\nOverall Results:")
        print(f"F1 Weighted: {results_dict['metrics']['f1_weighted']:.4f}")
        print(f"F1 Binary: {results_dict['metrics']['f1_binary']:.4f}")

    return results_dict

def make_predictions(model_path, test_data_path, output_path):
    """Make predictions using saved model"""
    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    # Make predictions (assumes test data has 'is_click' column for evaluation)
    X_test = test_df.drop(columns=['is_click'])
    predictions = model.predict(X_test)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'is_click': test_df['is_click'],
        'prediction': predictions,
    })

    if 'dataset_name' in test_df.columns:
        predictions_df['dataset_name'] = test_df['dataset_name']

    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return predictions_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results Analysis")
    parser.add_argument("-p", "--csv-path", default=DEFAULT_PREDICTION_CSV,
                        help="Path to predictions CSV file")
    parser.add_argument("-pd", "--per-dataset", action="store_true",
                        help="Calculate metrics per dataset")
    parser.add_argument("-m", "--model-path", default="models/model.joblib",
                        help="Path to saved model file")
    parser.add_argument("-t", "--test-data", default="data/test.csv",
                        help="Path to test data CSV (used to generate predictions if needed)")
    parser.add_argument("-o", "--output-dir", default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Check if the predictions CSV file exists; if not, prompt the user for a path.
    if not os.path.exists(args.csv_path):
        user_input = input(f"Predictions file not found at '{args.csv_path}'. Please provide the path to a predictions CSV file (or press Enter to generate predictions using the model): ").strip()
        if user_input:
            args.csv_path = user_input
        else:
            # If no path is provided, try generating predictions using the model (if available)
            if os.path.exists(args.model_path):
                print(f"Model found at {args.model_path}. Generating predictions using test data at {args.test_data}...")
                predictions_df = make_predictions(args.model_path, args.test_data, args.csv_path)
            else:
                print("No predictions file provided and no model available to generate predictions. Exiting.")
                exit(1)
    else:
        print(f"Loading predictions from {args.csv_path}")

    # If the predictions CSV still doesn't exist, generate predictions if a model is provided.
    if not os.path.exists(args.csv_path):
        if os.path.exists(args.model_path):
            print(f"Generating predictions using model at {args.model_path} and test data at {args.test_data}...")
            predictions_df = make_predictions(args.model_path, args.test_data, args.csv_path)
        else:
            print("Predictions file not found and model not available. Exiting.")
            exit(1)

    # Finally, calculate and save results
    results(args.csv_path, args.per_dataset, args.output_dir)
