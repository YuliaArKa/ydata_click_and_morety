import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import io
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, balanced_accuracy_score
)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
from src.preprocess import preprocess_data


# -------------------------
# Model Loader
# -------------------------
@st.cache_resource(show_spinner=False)
def get_model(model_path):
    # Check if the model file exists 
    if not Path(model_path).exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    # Check if model.joblib or model.pkl file
    if model_path.suffix == '.joblib':
         with open(model_path, 'rb') as f:
            model = joblib.load(f)
    elif model_path.suffix =='.pkl' or model_path.suffix =='.sav':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        st.error("Model file must be a .joblib or .pkl file.")
        st.stop()
    
    return model

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, X_test, y_test, threshold):
    """Evaluates the model using an adjustable threshold for classification
       and displays various Plotly charts including cool visualizations for
       classification metrics (with balanced accuracy) and model explainability."""
    # Get predicted probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    # Apply the threshold slider to generate class predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # -----------------------
    # Classification Metrics as Indicators
    # -----------------------
    fig_metrics = make_subplots(
        rows=1, cols=6,
        subplot_titles=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Balanced Accuracy"],
        specs=[[{"type": "indicator"}] * 6]
    )
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=acc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=1)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=prec * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=2)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=rec * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=3)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=f1 * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=4)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=auc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=5)
    
    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=balanced_acc * 100,
        number={'suffix': '%', 'valueformat': '.1f'}
    ), row=1, col=6)
    
    fig_metrics.update_layout(height=200, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_metrics)
    
    # -----------------------
    # ROC Curve
    # -----------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"AUC = {auc:.3f}",
                                 line=dict(color='darkorange')))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random',
                                 line=dict(dash='dash', color='navy')))
    fig_roc.update_layout(title="ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc)
    
    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_cm.update_layout(title="Confusion Matrix",
                         xaxis_title="Predicted Label",
                         yaxis_title="True Label")
    st.plotly_chart(fig_cm)
    
    # -----------------------
    # Classification Report
    # -----------------------
    st.text("Classification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
    
    # -----------------------
    # Histogram of Predicted Probabilities
    # -----------------------
    fig_hist = px.histogram(x=y_proba, nbins=50, title="Histogram of Predicted Probabilities")
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold = {threshold:.2f}")
    fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Frequency")
    st.plotly_chart(fig_hist)


# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="CTR Click Prediction Evaluation", layout="wide")
    st.title("CTR Click Prediction Evaluation App")

    # Define fixed paths for training (used in preprocessing) and evaluation test set.
    #TRAIN_PATH = "../data/train_dataset_full.csv"  # update if needed
    #EVAL_TEST_PATH = "../data/X_test_1st.csv"         # used for evaluation tab
    #LABEL_TEST_PATH = '../data/y_test_1st.csv'
    #MODEL_PATH = '../models/XGBoost.joblib'
    BASE_DIR = Path(__file__).resolve().parent.parent
    # Define fixed paths using the pathlib syntax
    TRAIN_PATH = BASE_DIR / "data" / "train_dataset_full.csv"
    EVAL_TEST_PATH = BASE_DIR / "data" / "X_test_1st.csv"
    LABEL_TEST_PATH = BASE_DIR / "data" / "y_test_1st.csv"
    MODEL_PATH = BASE_DIR / "models" / "RandomForest.joblib"
    # Create two tabs: one for uploading a test set & scoring, and one for model evaluation.
    tab_upload, tab_eval = st.tabs(["Upload & Score", "Model Evaluation"])
    
    # -------------------------
    # Tab 1: Upload & Score
    # -------------------------
    with tab_upload:
        st.header("Upload Test Set & Download Predictions")
        st.markdown("""
        **Features:**
        - Upload your test set (CSV file) from your local machine.
        - The app will produce predicted probabilities (scores) for each sample.
        - Download a CSV file containing only the scores, with no header, no index, and no extra columns.
        - The scores order matches the test set sample order.
        """)
        
        uploaded_test_file = st.file_uploader("Upload Test Set CSV", type=["csv"])
        if uploaded_test_file is not None:
            try:
                # Display a preview of the uploaded file.
                test_df_preview = pd.read_csv(uploaded_test_file)
                st.write("Preview of Uploaded Test Set:")
                st.dataframe(test_df_preview.head())
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
            
            # Write the uploaded file to a temporary file so that preprocessing.py can read it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_test_file.getvalue())
                tmp_path = tmp.name
            
            with st.spinner("Applying preprocessing..."):
                # Call preprocess_data in "predict" mode.
                # This will load the training set from TRAIN_PATH internally.
                x_test_dummies, _ = preprocess_data("predict", train_path=TRAIN_PATH, test_path=tmp_path)
            
            
            with st.spinner("Loading model and making predictions..."):
                model = get_model(MODEL_PATH)
                predicted_scores = model.predict_proba(x_test_dummies)[:, 1]
            
            st.success("Predictions completed!")
            
            # Prepare CSV output: one column of scores, no header, no index.
            output_csv = io.StringIO()
            pd.DataFrame(predicted_scores).to_csv(output_csv, header=False, index=False)
            csv_output = output_csv.getvalue()
            
            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Please upload a test set CSV file.")
    
    # -------------------------
    # Tab 2: Model Evaluation
    # -------------------------
    with tab_eval:
        st.subheader("Preprocessing & Evaluation")
        with st.spinner("Loading and preprocessing data..."):
            # For evaluation, we load our internal test set and its labels.
            y_test = pd.read_csv(LABEL_TEST_PATH, header=None).values.reshape(-1)
            # Call preprocess_data in "predict" mode using fixed file paths.
            x_test_dummies, _ = preprocess_data("predict", train_path=TRAIN_PATH, test_path=EVAL_TEST_PATH)
            st.dataframe(x_test_dummies.head())
        st.success("Data loaded and preprocessed successfully!")
        
        with st.spinner("Loading model..."):
            model = get_model(MODEL_PATH)
        st.success("Model loaded!")

        # Sidebar options for evaluation.
        threshold = st.sidebar.slider("Probability Threshold", min_value=0.0, max_value=1.0,
                                      value=0.5, step=0.01)
        show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
        
        # Evaluate the model.
        evaluate_model(model, x_test_dummies, y_test, threshold)
        
        # Feature Importance (if available).
        if show_feature_importance and hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            feature_names = x_test_dummies.columns
            df_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df_importance.sort_values("Importance", ascending=True, inplace=True)
            fig_imp = px.bar(df_importance, x="Importance", y="Feature", orientation="h",
                             title="Feature Importances")
            st.plotly_chart(fig_imp)
        
        
if __name__ == '__main__':
    main()
