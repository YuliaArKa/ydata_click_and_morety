from invoke import task
import os
from datetime import datetime
import glob

# Default configurations
DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PRED_FILE = "predictions.csv"

# Model configurations with their specific parameters and defaults
MODEL_CONFIGS = {
    'lightgbm': {
        'script': 'train_lgb.py',
        'params': {
            'num_leaves': ('--num-leaves', 31),
            'learning_rate': ('--learning-rate', 0.1),
            'n_estimators': ('--n-estimators', 100)
        }
    },
    'xgboost': {
        'script': 'train_xgb.py',
        'params': {
            'max_depth': ('--max-depth', 6),
            'learning_rate': ('--learning-rate', 0.1),
            'n_estimators': ('--n-estimators', 100)
        }
    },
    'decision_tree': {
        'script': 'train_dt.py',
        'params': {
            'max_depth': ('--max-depth', 10),
            'min_samples_split': ('--min-samples-split', 2),
            'min_samples_leaf': ('--min-samples-leaf', 1)
        }
    },
    'logistic': {
        'script': 'train_logistic.py',
        'params': {
            'max_iter': ('--max-iter', 100),
            'C': ('--C', 1.0)
        }
    },
    'adaboost': {
        'script': 'train_adaboost.py',
        'params': {
            'n_estimators': ('--n-estimators', 100),
            'learning_rate': ('--learning-rate', 0.1)
        }
    }
}


def get_available_models():
    """Get list of available models based on existing training scripts"""
    available_models = []
    for model_type, config in MODEL_CONFIGS.items():
        if os.path.exists(config['script']):
            available_models.append(model_type)
    return available_models


@task
def list_models(ctx):
    """List all available models and their parameters"""
    available_models = get_available_models()
    print("\nAvailable models and their parameters:")
    for model in available_models:
        print(f"\n{model.upper()}:")
        for param, (flag, default) in MODEL_CONFIGS[model]['params'].items():
            print(f"  {param}: {flag} (default: {default})")


@task
def preprocess(ctx, mode="train",
               train_path=f"{DEFAULT_DATA_DIR}/train_dataset_full.csv",
               test_path=None):
    """
    Preprocess data.

    For training: mode 'train' and provide train_path.
    For prediction: mode 'predict' and provide both train_path and test_path.
    """
    cmd = f"python preprocess.py --mode {mode} --train_path {train_path}"
    if mode == "predict":
        if test_path is None:
            print("Error: test_path must be provided for prediction mode.")
            return
        cmd += f" --test_path {test_path}"
    print(f"Running: {cmd}")
    ctx.run(cmd)


@task
def train(ctx, model_type="all", model_dir=DEFAULT_MODELS_DIR, **kwargs):
    """
    Train the specified model(s) with custom parameters.

    Example usages:
      invoke train --model-type decision_tree --max_depth 15 --min_samples_split 5
      invoke train --model-type lightgbm --num_leaves 63 --learning_rate 0.05
    """
    available_models = get_available_models()
    if not available_models:
        raise ValueError("No training scripts found!")
    if model_type.lower() not in ["all"] + available_models:
        raise ValueError(f"Invalid model type. Available options: {['all'] + available_models}")

    print(f"Running training step for {model_type}...")
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_to_train = available_models if model_type.lower() == "all" else [model_type.lower()]

    for model in models_to_train:
        print(f"\nTraining {model} model...")
        config = MODEL_CONFIGS[model]
        cmd_params = []
        for param_name, (flag, default) in config['params'].items():
            # Use provided value if available; otherwise, use default.
            value = kwargs.get(param_name, default)
            cmd_params.append(f"{flag} {value}")
        cmd = (
            f"python {config['script']} "
            f"--model-name {model}_{timestamp} "
            f"{' '.join(cmd_params)} "
            f"--output-dir {model_dir}"
        )
        print(f"Running command: {cmd}")
        ctx.run(cmd)


@task
def predict(ctx, model="NaiveBayes",
            output_csv_path=f"{DEFAULT_RESULTS_DIR}/predictions.csv",
            data_path=f"{DEFAULT_DATA_DIR}/X_test_1st.csv",
            preprocess_flag=False,
            train_path=f"{DEFAULT_DATA_DIR}/train_dataset_full.csv",
            raw_data=f"{DEFAULT_DATA_DIR}/X_test_1st.csv",
            model_dir=DEFAULT_MODELS_DIR):
    """
    Run prediction using predict.py.

    If preprocess_flag is set, prediction will run with preprocessing.
    """
    cmd = f"python predict.py -b {model} -o {output_csv_path} -d {data_path}"
    if preprocess_flag:
        cmd += f" -p --raw-data {raw_data} --train-path {train_path}"
    cmd += f" -model-dir {model_dir}"
    print(f"Running: {cmd}")
    ctx.run(cmd)


@task
def analyze(ctx,
            csv_path=DEFAULT_PRED_FILE,
            per_dataset=False,
            model_path=f"{DEFAULT_MODELS_DIR}/model.joblib",
            test_data=f"{DEFAULT_DATA_DIR}/test.csv",
            output_dir=DEFAULT_RESULTS_DIR):
    """
    Analyze predictions using results.py.
    """
    cmd = f"python results.py -p {csv_path} -o {output_dir} -t {test_data} -m {model_path}"
    if per_dataset:
        cmd += " -pd"
    print(f"Running: {cmd}")
    ctx.run(cmd)


@task
def archive(ctx, name, base_folder="archived_experiments"):
    """
    Archive experiment directories (data, results, models).
    """
    DATE_TIME_PATTERN = "%Y%m%d_%H%M%S"
    name = f"{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}"
    print(f"Archiving experiment: {name}")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    exp_path = os.path.join(base_folder, name)
    ctx.run(f"mkdir {exp_path}")
    for folder in ["data", "results", "models"]:
        if os.path.exists(folder):
            ctx.run(f"mv {folder} {exp_path}")
            ctx.run(f"mkdir {folder}")
    print(f"Archived experiment to {exp_path}")


@task
def clean(ctx):
    """
    Clean up generated files in the models and results directories.
    """
    for folder in [DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR]:
        if os.path.exists(folder):
            print(f"Cleaning {folder}...")
            files = glob.glob(os.path.join(folder, "*"))
            for f in files:
                os.remove(f)
    print("Clean complete.")


@task
def run_pipeline(ctx, model_type="all", per_dataset=False, **kwargs):
    """
    Run the complete pipeline:
      1. Preprocess training data
      2. Train model(s)
      3. Predict on test data
      4. Analyze results

    Example:
      invoke run-pipeline --model-type decision_tree --max_depth 15 --min_samples_split 5
    """
    print("Running complete pipeline...")

    # Step 1: Preprocess training data
    preprocess_cmd = f"python preprocess.py --mode train --train_path {DEFAULT_DATA_DIR}/train_dataset_full.csv"
    print(f"Running: {preprocess_cmd}")
    ctx.run(preprocess_cmd)

    # Step 2: Train model(s)
    train(ctx, model_type=model_type, **kwargs)

    # Step 3: Predict using (for example) the NaiveBayes model.
    predict_cmd = (
        f"python predict.py -b NaiveBayes -o {DEFAULT_RESULTS_DIR}/predictions.csv "
        f"-d {DEFAULT_DATA_DIR}/X_test_1st.csv -p --raw-data {DEFAULT_DATA_DIR}/X_test_1st.csv "
        f"--train-path {DEFAULT_DATA_DIR}/train_dataset_full.csv -model-dir {DEFAULT_MODELS_DIR}"
    )
    print(f"Running: {predict_cmd}")
    ctx.run(predict_cmd)

    # Step 4: Analyze predictions
    analyze(ctx, csv_path=f"{DEFAULT_RESULTS_DIR}/predictions.csv", per_dataset=per_dataset)

    print("Pipeline execution complete.")
