README

# YData Click and More Pipeline

This repository contains a complete data processing and machine learning pipeline for the **YData Click and More** project. The pipeline includes the following steps:

1. **Preprocessing:** Prepare raw data for training or prediction.
2. **Training:** Train multiple models with various algorithms.
3. **Prediction:** Generate predictions using a saved model (or a stacking ensemble).
4. **Analysis:** Compute classification metrics and generate reports/plots.
5. **Archiving:** Archive experiments (data, results, and models).
6. **Cleaning:** Remove generated files from models and results directories.
7. **Pipeline:** Run the entire end-to-end pipeline with a single command.

This project is orchestrated using [Invoke](http://www.pyinvoke.org/), which allows you to run tasks from the command line.

---

## Prerequisites

- **Python 3.7+**
- Install required packages via pip. For example:

  ```bash
  pip install pandas numpy scikit-learn seaborn matplotlib joblib wandb invoke

    Ensure that the following scripts are present in the repository root:
        preprocess.py
        train_dt.py, train_lgb.py, train_xgb.py, train_logistic.py, train_adaboost.py, train_rf.py, train_nb.py, train_knn.py
        predict.py
        results.py
        tasks.py

Repository Structure

.
├── data
│   ├── train_dataset_full.csv       # Raw training data
│   ├── test.csv                     # Preprocessed test split (from training)
│   └── X_test_1st.csv               # Raw prediction data
├── models                           # Directory where trained models are saved
├── results                          # Directory where prediction and analysis results are saved
├── archived_experiments             # Directory for archived experiments
├── src
│   ├── preprocess.py                # Data preprocessing script
│   ├── train_dt.py                  # Decision Tree training script
│   ├── train_lgb.py                 # LightGBM training script
│   ├── train_xgb.py                 # XGBoost training script
│   ├── train_logistic.py            # Logistic Regression training script
│   ├── train_adaboost.py            # AdaBoost training script
│   ├── train_rf.py                  # Random Forest training script
│   ├── train_nb.py                  # Naive Bayes training script
│   ├── train_knn.py                 # KNN training script
│   ├── predict.py                   # Prediction script
│   ├── results.py                   # Analysis and results generation script
│   └── tasks.py                     # Invoke tasks to run the full pipeline

Usage

All tasks are defined in tasks.py using Invoke. To use these tasks, first install Invoke (if not already installed):

pip install invoke

Then, from the repository root, you can run the following commands:
List Available Models

To list the models available for training and their parameters:

invoke list-models

Preprocessing
For Training:

invoke preprocess --mode train --train_path data/train_dataset_full.csv

This will preprocess your training data and generate train.csv and test.csv.
For Prediction:

invoke preprocess --mode predict --train_path data/train_dataset_full.csv --test_path data/X_test_1st.csv

This will preprocess your prediction data and save it as predict.csv.
Training Models

You can train one or more models. For example, to train a Decision Tree with custom parameters:

invoke train --model-type decision_tree --max_depth 15 --min_samples_split 5

Or, to train all available models with default parameters:

invoke train --model-type all

Model Training Parameters

Below are the supported models and their parameters (with example defaults):

    AdaBoost (script: train_adaboost.py)
        --n-estimators (default: 100)
        --learning-rate (default: 0.1)

    LightGBM (script: train_lgb.py)
        --num-leaves (default: 31)
        --learning-rate (default: 0.1)
        --n-estimators (default: 100)
        --max-depth (default: 10)
        --colsample-bytree (default: 1.0)
        --subsample (default: 1.0)
        --min-child-samples (default: 20)
        --reg-alpha (default: 0.0)
        --reg-lambda (default: 0.0)
        --min-split-gain (default: 0.0)
        --feature-fraction (default: 1.0)
        --bagging-fraction (default: 1.0)
        --bagging-freq (default: 0)
        --metric (default: "binary_logloss")

    XGBoost (script: train_xgb.py)
        --max-depth (default: 6)
        --learning-rate (default: 0.1)
        --n-estimators (default: 100)
        --subsample (default: 1.0)
        --colsample-bytree (default: 1.0)
        --scale-pos-weight (default: 1.0)
        --gamma (default: 0.0)
        --min-child-weight (default: 1)
        --reg-alpha (default: 0.0)
        --reg-lambda (default: 1.0)
        --eval-metric (default: "logloss")

    Logistic Regression (script: train_logistic.py)
        --C (default: 1.0)
        --max-iter (default: 100)
        --solver (default: "lbfgs")
        --penalty (default: "l2")

    Decision Tree (script: train_dt.py)
        --max-depth (default: 10)
        --min-samples-split (default: 2)
        --min-samples-leaf (default: 1)

    Random Forest (script: train_rf.py)
        --n-estimators (default: 100)
        --max-depth (default: 10)
        --min-samples-split (default: 2)
        --min-samples-leaf (default: 1)
        --max-features (default: "auto")
        --bootstrap (default: True)
        --min-weight-fraction-leaf (default: 0.0)
        --max-leaf-nodes (default: None)
        --criterion (default: "gini")

    Naive Bayes (script: train_nb.py)
        --var-smoothing (default: 1e-9)

    KNN (script: train_knn.py)
        --n-neighbors (default: 5)
        --weights (default: "uniform")
        --algorithm (default: "auto")
        --leaf-size (default: 30)

Prediction

Run predictions using a specified model. For example, to run prediction using the NaiveBayes model with preprocessing enabled:

invoke predict --model NaiveBayes --output-csv-path results/predictions.csv --preprocess

Parameters:

    -b or --model: Base model name (or "stacking" for ensemble).
    -o or --output-csv-path: Full path (including filename) for the predictions CSV.
    -d or --data-path: Path to the input data CSV.
    -p or --preprocess: Enable preprocessing before prediction.
    -r or --raw-data: Raw data path (required if preprocessing is enabled).
    --train-path: Path to training data (required if preprocessing is enabled).
    -model-dir: Directory containing trained models.

Analysis

Analyze the prediction results and generate classification reports:

invoke analyze --csv-path results/predictions.csv --per-dataset

Parameters:

    -p or --csv-path: Path to the predictions CSV.
    -pd or --per-dataset: Calculate metrics per dataset (if your predictions CSV contains a dataset_name column).
    -m or --model-path: Path to the saved model file.
    -t or --test-data: Path to the test data CSV (for generating predictions if needed).
    -o or --output-dir: Directory to save results.

Archive Experiments

Archive your current experiment directories (data, results, models) into a timestamped folder:

invoke archive --name "my_experiment"

Clean Up

Remove generated files from the models and results directories:

invoke clean

Run the Entire Pipeline

Run all steps (preprocessing, training, prediction, analysis) in one command. For example, to run the pipeline with a Decision Tree:

invoke run-pipeline --model-type decision_tree --max_depth 15 --min_samples_split 5

This command:

    Preprocesses the training data.
    Trains the specified model(s).
    Generates predictions (using, for example, the NaiveBayes model).
    Analyzes the predictions and produces a classification report.

Additional Notes

    Individual Scripts:
    Each script (e.g., preprocess.py, train_*.py, predict.py, results.py) can also be run standalone.
    Customizing Defaults:
    You can modify default directories (data, models, results) by editing the constants at the top of tasks.py.
    Invoke Documentation:
    For more information on using Invoke, please refer to Invoke’s official documentation.

License

MIT License