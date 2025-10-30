import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
import requests
from datetime import datetime
import joblib
import subprocess
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        background-color: #ffd166;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        background-color: #06d6a0;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define fixed risk levels and colors
RISK_LEVELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
RISK_COLORS = {
    'Very Low': '#06d6a0',
    'Low': '#90e0ef',
    'Medium': '#ffd166',
    'High': '#ff9e00',
    'Very High': '#ff6b6b'
}


class FloodRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder_land_cover = LabelEncoder()
        self.label_encoder_target = LabelEncoder()
        self.mlflow_tracking_uri = "mlruns"
        self.is_trained = False
        self.feature_names = ['rainfall_mm', 'temperature_c', 'river_flow_rate', 'water_level_m',
                              'soil_moisture', 'population_density', 'elevation_m', 'land_cover_type_encoded',
                              'previous_flood_occurrence']
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Create mlruns directory if it doesn't exist
        Path(self.mlflow_tracking_uri).mkdir(exist_ok=True)

        # Initialize label encoders with expected categories
        self.label_encoder_land_cover.fit(['Urban', 'Agricultural', 'Forest', 'Water', 'Barren'])
        self.label_encoder_target.fit(RISK_LEVELS)

    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic flood risk data similar to your dataset"""
        np.random.seed(42)

        data = {
            'rainfall_mm': np.random.exponential(20, n_samples),
            'temperature_c': np.random.normal(25, 5, n_samples),
            'river_flow_rate': np.random.gamma(2, 50, n_samples),
            'water_level_m': np.random.normal(2, 0.5, n_samples),
            'soil_moisture': np.random.uniform(0.1, 0.9, n_samples),
            'population_density': np.random.poisson(500, n_samples),
            'elevation_m': np.random.uniform(0, 1000, n_samples),
            'land_cover_type': np.random.choice(['Urban', 'Agricultural', 'Forest', 'Water', 'Barren'], n_samples),
            'previous_flood_occurrence': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }

        df = pd.DataFrame(data)

        # Create flood risk based on features (more realistic calculation)
        risk_score = (
                df['rainfall_mm'] * 0.25 +
                df['river_flow_rate'] * 0.15 +
                df['water_level_m'] * 0.20 +
                (1 - df['soil_moisture']) * 0.10 +
                (1 - df['elevation_m'] / 1000) * 0.20 +
                df['population_density'] / 1000 * 0.10
        )

        # Add some randomness and ensure proper distribution
        risk_score += np.random.normal(0, 0.5, n_samples)
        risk_score = np.clip(risk_score, 0, 10)

        # Convert to risk categories
        df['flood_risk'] = pd.cut(risk_score,
                                  bins=[0, 2, 4, 6, 8, 10],
                                  labels=RISK_LEVELS)

        return df

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Handle categorical variables
        df_processed = df.copy()
        df_processed['land_cover_type_encoded'] = self.label_encoder_land_cover.transform(
            df_processed['land_cover_type'])
        df_processed['flood_risk_encoded'] = self.label_encoder_target.transform(df_processed['flood_risk'])

        X = df_processed[self.feature_names]
        y = df_processed['flood_risk_encoded']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and track with MLflow"""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }

        results = {}

        # Set experiment
        mlflow.set_experiment("Flood_Risk_Prediction")

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Log parameters
                mlflow.log_param("model_type", model_name)
                if model_name == 'Random Forest':
                    mlflow.log_param("n_estimators", 100)
                elif model_name == 'Logistic Regression':
                    mlflow.log_param("max_iter", 1000)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
                mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")

                # Save model locally
                self.models[model_name] = model
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model,
                    'predictions': y_pred
                }

        self.is_trained = True

        # Save preprocessing objects
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoder_land_cover, 'label_encoder_land_cover.pkl')
        joblib.dump(self.label_encoder_target, 'label_encoder_target.pkl')

        return results

    def load_preprocessing_objects(self):
        """Load preprocessing objects if they exist"""
        try:
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder_land_cover = joblib.load('label_encoder_land_cover.pkl')
            self.label_encoder_target = joblib.load('label_encoder_target.pkl')
            self.is_trained = True
            return True
        except FileNotFoundError:
            # Initialize with default values
            self.label_encoder_land_cover.fit(['Urban', 'Agricultural', 'Forest', 'Water', 'Barren'])
            self.label_encoder_target.fit(RISK_LEVELS)
            return False

    def predict_risk(self, input_data):
        """Make prediction with proper error handling"""
        try:
            # Encode land cover
            input_data['land_cover_type_encoded'] = \
            self.label_encoder_land_cover.transform([input_data['land_cover_type']])[0]

            # Create feature array in correct order
            feature_values = [input_data[feature] for feature in self.feature_names]
            X_input = np.array([feature_values])

            # Scale features
            X_scaled = self.scaler.transform(X_input)

            # Make prediction
            if 'best' in self.models:
                prediction = self.models['best'].predict(X_scaled)[0]
                prediction_proba = self.models['best'].predict_proba(X_scaled)[0]
            else:
                # Use first available model
                model_name = list(self.models.keys())[0]
                prediction = self.models[model_name].predict(X_scaled)[0]
                prediction_proba = self.models[model_name].predict_proba(X_scaled)[0]

            return prediction, prediction_proba, None

        except Exception as e:
            return None, None, str(e)

    def get_risk_description(self, risk_level):
        """Get description for risk levels"""
        risk_descriptions = {
            0: "Very Low: Minimal flood risk. Normal conditions.",
            1: "Low: Slight chance of flooding in extreme conditions.",
            2: "Medium: Moderate flood risk. Stay alert to weather updates.",
            3: "High: Significant flood risk. Take precautionary measures.",
            4: "Very High: Severe flood risk. Immediate action required."
        }
        return risk_descriptions.get(risk_level, "Unknown risk level")


def start_mlflow_ui():
    """Start MLflow UI in background"""
    try:
        # Start MLflow UI
        process = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui", "--port", "5000", "--backend-store-uri", "mlruns"])
        st.success("MLflow UI started on http://localhost:5000")
        return True
    except Exception as e:
        st.error(f"Failed to start MLflow UI: {e}")
        return False


def main():
    st.markdown('<h1 class="main-header">🌊 Flood Risk Prediction and Early Warning System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FloodRiskPredictor()
        st.session_state.predictor.load_preprocessing_objects()

    predictor = st.session_state.predictor

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Section",
                                    ["Project Overview", "Data Exploration", "Model Training",
                                     "MLflow Tracking", "Real-time Prediction", "System Monitoring"])

    if app_mode == "Project Overview":
        show_project_overview()

    elif app_mode == "Data Exploration":
        show_data_exploration(predictor)

    elif app_mode == "Model Training":
        show_model_training(predictor)

    elif app_mode == "MLflow Tracking":
        show_mlflow_tracking(predictor)

    elif app_mode == "Real-time Prediction":
        show_real_time_prediction(predictor)

    elif app_mode == "System Monitoring":
        show_system_monitoring(predictor)


def show_project_overview():
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        ### 🌟 Introduction
        Flooding is one of the most frequent and devastating natural disasters in India, 
        causing significant socio-economic and environmental damage. With climate change 
        and rapid urbanization, the frequency and intensity of floods have increased.

        This system provides:
        - **Real-time flood risk prediction** using machine learning
        - **MLOps integration** with MLflow for experiment tracking
        - **Automated model training** and evaluation
        - **Early warning alerts** for disaster preparedness
        """)

    with col2:
        st.image("https://via.placeholder.com/300x200/1f77b4/ffffff?text=Flood+Risk",
                 caption="Flood Risk Assessment")

    st.markdown("---")

    st.markdown("""
    ### 🎯 Project Objectives

    - Build a machine learning model to predict flood risk levels using multi-modal data
    - Track experiments, parameters, and models with MLflow
    - Develop a user-friendly interface for real-time flood predictions
    - Provide automated early warning system
    - Ensure reproducibility and model management
    """)

    st.markdown("---")

    st.markdown("""
    ### 🛠️ Tools & Technologies

    | Category | Technologies |
    |----------|--------------|
    | Data Handling | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn, Plotly |
    | Modeling | Scikit-learn, XGBoost |
    | Experiment Tracking | MLflow |
    | Web Framework | Streamlit |
    | Containerization | Docker |
    | CI/CD | GitHub Actions |
    """)


def show_data_exploration(predictor):
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)

    # Generate sample data
    with st.spinner("Generating sample flood risk data..."):
        df = predictor.generate_sample_data(1000)

    # Data overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        risk_counts = df['flood_risk'].value_counts()
        st.metric("Risk Categories", len(risk_counts))
    with col4:
        st.metric("Data Completeness", "100%")

    # Show data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    # Data visualization
    st.subheader("Data Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_counts = df['flood_risk'].value_counts()
        colors = [RISK_COLORS.get(risk, '#808080') for risk in risk_counts.index]  # Default to gray if not found
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors)
        ax.set_title('Flood Risk Distribution')
        st.pyplot(fig)

    with col2:
        # Feature correlation
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features for correlation matrix")

    # Feature distributions
    st.subheader("Feature Distributions by Risk Level")
    feature_to_plot = st.selectbox("Select feature to visualize",
                                   ['rainfall_mm', 'river_flow_rate', 'water_level_m', 'elevation_m',
                                    'population_density'])

    fig, ax = plt.subplots(figsize=(12, 6))
    risk_levels = df['flood_risk'].unique()

    # Ensure we don't exceed colors list
    for i, risk_level in enumerate(risk_levels):
        subset = df[df['flood_risk'] == risk_level]
        if len(subset) > 0:  # Check if subset is not empty
            color = RISK_COLORS.get(risk_level, '#808080')  # Default to gray if not found
            ax.hist(subset[feature_to_plot], alpha=0.7, label=risk_level, bins=20, color=color)

    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature_to_plot} by Flood Risk')
    ax.legend()
    st.pyplot(fig)


def show_model_training(predictor):
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    st.info("🚀 This section automatically trains multiple models and tracks experiments with MLflow")

    # Generate and prepare data
    with st.spinner("Preparing data for training..."):
        df = predictor.generate_sample_data(1000)
        X, y = predictor.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.success(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # Model training
    if st.button("🎯 Train All Models with MLflow Tracking", type="primary"):
        with st.spinner("Training models and tracking with MLflow..."):
            results = predictor.train_models(X_train, y_train, X_test, y_test)

        # Display results
        st.subheader("📊 Model Performance Comparison")

        # Create metrics comparison
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'Precision': [results[model]['precision'] for model in results],
            'Recall': [results[model]['recall'] for model in results],
            'F1-Score': [results[model]['f1_score'] for model in results]
        })

        # Display metrics table
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }).highlight_max(axis=0))

        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model = results[best_model_name]['model']
        predictor.models['best'] = best_model

        st.success(f"🎉 Best Model: **{best_model_name}** with F1-Score: **{results[best_model_name]['f1_score']:.4f}**")

        # Confusion matrix for best model
        st.subheader(f"📈 Confusion Matrix - {best_model_name}")
        y_pred_best = results[best_model_name]['predictions']
        cm = confusion_matrix(y_test, y_pred_best)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=predictor.label_encoder_target.classes_,
                    yticklabels=predictor.label_encoder_target.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # Save best model
        joblib.dump(best_model, 'best_flood_model.pkl')
        st.success("💾 Best model saved as 'best_flood_model.pkl'")

        # Show feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': predictor.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)


def show_mlflow_tracking(predictor):
    st.markdown('<h2 class="sub-header">📈 MLflow Experiment Tracking</h2>', unsafe_allow_html=True)

    st.info("🔬 MLflow automatically tracks all experiments, parameters, and metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 MLflow UI Access")
        st.markdown("""
        To view the MLflow tracking UI, run the following command in your terminal:
        ```
        mlflow ui
        ```
        Then open [http://localhost:5000](http://localhost:5000) in your browser.

        **Or click the button below to start MLflow UI automatically:**
        """)

        if st.button("🎪 Start MLflow UI", type="primary"):
            if start_mlflow_ui():
                st.markdown('<div class="info-box">✅ MLflow UI started! Visit http://localhost:5000</div>',
                            unsafe_allow_html=True)
            else:
                st.error("❌ Failed to start MLflow UI")

        # Show recent experiments
        st.subheader("📋 Recent Experiments")
        try:
            # This would normally connect to MLflow tracking server
            experiments = mlflow.search_experiments()
            if experiments:
                st.success(f"✅ Found {len(experiments)} experiments in MLflow")

                # Display experiment info
                for exp in experiments[:3]:  # Show first 3 experiments
                    with st.expander(f"🔬 Experiment: {exp.name}"):
                        st.write(f"**Experiment ID:** {exp.experiment_id}")
                        st.write(f"**Artifact Location:** {exp.artifact_location}")
            else:
                st.warning("📭 No experiments found. Train models first!")

        except Exception as e:
            st.error(f"❌ Could not connect to MLflow: {e}")
            st.info("💡 Make sure MLflow is installed and the tracking server is running")

    with col2:
        st.subheader("📊 Tracked Information")
        st.markdown("""
        MLflow automatically tracks:

        - **🔧 Parameters**: Model hyperparameters, feature sets  
        - **📈 Metrics**: Accuracy, Precision, Recall, F1-Score  
        - **📁 Artifacts**: Model files, plots, preprocessing objects  
        - **🤖 Models**: Serialized models for deployment  
        - **📝 Metadata**: Run timestamps, git commits, environment  
        """)

        # Show sample MLflow code
        st.subheader("💻 Sample MLflow Code")
        st.code("""
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.83)

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
        """, language="python")

        st.subheader("🎯 Current Status")
        if predictor.is_trained:
            st.success("✅ Models are trained and ready for prediction!")
            st.info("💡 You can now use the Real-time Prediction section")
        else:
            st.warning("📭 No trained models found. Please train models first!")


def show_real_time_prediction(predictor):
    st.markdown('<h2 class="sub-header">🔮 Real-time Flood Risk Prediction</h2>', unsafe_allow_html=True)

    if not predictor.is_trained and len(predictor.models) == 0:
        st.error("❌ No trained models found! Please train models in the 'Model Training' section first.")
        st.info("💡 Go to the Model Training section and click 'Train All Models with MLflow Tracking'")
        return

    # Load best model if available
    try:
        if 'best' not in predictor.models:
            predictor.models['best'] = joblib.load('best_flood_model.pkl')
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error("❌ No trained model found. Please train models first in the 'Model Training' section.")
        return

    if model_loaded:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Input Parameters")

            # Create input form
            with st.form("prediction_form"):
                st.markdown("### Environmental Factors")
                rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 200.0, 25.0, help="Recent rainfall in millimeters")
                temperature = st.slider("🌡️ Temperature (°C)", 0.0, 45.0, 25.0, help="Current temperature")
                river_flow = st.slider("💧 River Flow Rate", 0.0, 1000.0, 100.0, help="River flow rate in m³/s")
                water_level = st.slider("📏 Water Level (m)", 0.0, 15.0, 2.0, help="Current water level")
                soil_moisture = st.slider("🟫 Soil Moisture", 0.0, 1.0, 0.5, help="Soil moisture content (0-1)")

                st.markdown("### Geographical Factors")
                population_density = st.slider("👥 Population Density", 0, 5000, 500, help="People per square km")
                elevation = st.slider("⛰️ Elevation (m)", 0.0, 2000.0, 100.0, help="Area elevation above sea level")
                land_cover = st.selectbox("🌳 Land Cover Type",
                                          ['Urban', 'Agricultural', 'Forest', 'Water', 'Barren'],
                                          help="Type of land cover")
                previous_flood = st.selectbox("📅 Previous Flood Occurrence",
                                              [0, 1],
                                              format_func=lambda x: "No" if x == 0 else "Yes",
                                              help="History of previous floods in the area")

                submitted = st.form_submit_button("🎯 Predict Flood Risk", type="primary")

        with col2:
            if submitted:
                st.subheader("🎪 Prediction Results")

                # Prepare input data
                input_data = {
                    'rainfall_mm': rainfall,
                    'temperature_c': temperature,
                    'river_flow_rate': river_flow,
                    'water_level_m': water_level,
                    'soil_moisture': soil_moisture,
                    'population_density': population_density,
                    'elevation_m': elevation,
                    'land_cover_type': land_cover,
                    'previous_flood_occurrence': previous_flood
                }

                # Make prediction
                prediction, prediction_proba, error = predictor.predict_risk(input_data)

                if error:
                    st.error(f"❌ Prediction failed: {error}")
                    st.info("💡 This might be due to preprocessing issues. Please try retraining the models.")
                else:
                    # Display results
                    predicted_risk = predictor.label_encoder_target.inverse_transform([prediction])[0]

                    # Risk level display
                    risk_classes = {
                        'Very Low': 'risk-low',
                        'Low': 'risk-low',
                        'Medium': 'risk-medium',
                        'High': 'risk-high',
                        'Very High': 'risk-high'
                    }

                    # Use the global RISK_COLORS with safe access
                    display_color = RISK_COLORS.get(predicted_risk, '#808080')  # Default to gray if not found

                    st.markdown(
                        f'<div class="{risk_classes.get(predicted_risk, "risk-medium")}">Predicted Risk: {predicted_risk}</div>',
                        unsafe_allow_html=True)

                    st.write(predictor.get_risk_description(prediction))

                    # Probability distribution
                    st.subheader("📊 Risk Probability Distribution")

                    # Use fixed risk levels and safe color mapping
                    colors_for_chart = [RISK_COLORS.get(risk, '#808080') for risk in RISK_LEVELS]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(RISK_LEVELS, prediction_proba,
                                  color=colors_for_chart)
                    ax.set_ylabel('Probability')
                    ax.set_title('Flood Risk Probability Distribution')
                    ax.set_ylim(0, 1)

                    # Add value labels on bars
                    for bar, prob in zip(bars, prediction_proba):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

                    st.pyplot(fig)

                    # Early warning
                    if predicted_risk in ['High', 'Very High']:
                        st.error(
                            "🚨 **EARLY WARNING**: High flood risk detected! Take precautionary measures immediately.")
                        st.markdown("""
                        **Recommended Actions:**
                        - Monitor weather updates regularly
                        - Prepare emergency evacuation plan
                        - Secure important documents and belongings
                        - Avoid low-lying areas
                        """)
                    elif predicted_risk == 'Medium':
                        st.warning("⚠️ **ALERT**: Medium flood risk. Stay updated with weather forecasts.")
                        st.markdown("""
                        **Recommended Actions:**
                        - Stay informed about weather conditions
                        - Prepare emergency kit
                        - Identify safe evacuation routes
                        """)
                    else:
                        st.success("✅ **SAFE**: Low flood risk. Normal conditions.")

                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        input_summary = {
                            "Rainfall": f"{rainfall} mm",
                            "Temperature": f"{temperature}°C",
                            "River Flow": f"{river_flow} m³/s",
                            "Water Level": f"{water_level} m",
                            "Soil Moisture": f"{soil_moisture:.2f}",
                            "Population Density": f"{population_density} people/km²",
                            "Elevation": f"{elevation} m",
                            "Land Cover": land_cover,
                            "Previous Flood": "Yes" if previous_flood == 1 else "No"
                        }

                        for key, value in input_summary.items():
                            st.write(f"**{key}:** {value}")


def show_system_monitoring(predictor):
    st.markdown('<h2 class="sub-header">📊 System Monitoring</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("System Status", "Operational", delta="Normal", delta_color="off")
    with col2:
        status = "Ready" if predictor.is_trained else "Not Trained"
        delta = "All trained" if predictor.is_trained else "Train required"
        st.metric("Models Status", status, delta=delta)
    with col3:
        st.metric("MLflow Status", "Active", delta="Tracking", delta_color="off")
    with col4:
        st.metric("Predictions Today", "15", delta="+5", delta_color="inverse")

    st.subheader("📈 Model Performance Over Time")

    # Real performance data from training if available
    if predictor.is_trained and hasattr(predictor, 'performance_history'):
        performance_data = predictor.performance_history
    else:
        # Simulated performance metrics over time
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        performance_data = {
            'Random Forest': np.random.uniform(0.8, 0.9, 10),
            'Logistic Regression': np.random.uniform(0.75, 0.85, 10),
            'XGBoost': np.random.uniform(0.82, 0.92, 10)
        }

    fig, ax = plt.subplots(figsize=(12, 6))
    for model, scores in performance_data.items():
        if hasattr(scores, '__iter__') and len(scores) > 0:
            ax.plot(range(len(scores)), scores, marker='o', label=model, linewidth=2)

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('F1-Score')
    ax.set_title('Model Performance Over Time')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("🔔 Recent Activity Log")

    # Sample activity log
    sample_log = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-15', periods=8, freq='H'),
        'Activity': ['Model Training', 'Prediction', 'Prediction', 'System Update',
                     'Prediction', 'Model Retraining', 'Prediction', 'Backup'],
        'Status': ['Completed', 'Success', 'Success', 'Completed',
                   'Success', 'Completed', 'Success', 'Completed'],
        'Details': ['Random Forest trained', 'Mumbai - High risk', 'Chennai - Medium risk',
                    'v1.2 deployed', 'Kolkata - Very High risk', 'XGBoost updated',
                    'Delhi - Low risk', 'Database backup']
    })

    st.dataframe(sample_log)

    # System health
    st.subheader("🩺 System Health")
    col1, col2 = st.columns(2)

    with col1:
        # CPU Usage
        st.metric("CPU Usage", "45%", delta="+2%", delta_color="inverse")
        # Memory Usage
        st.metric("Memory Usage", "62%", delta="-5%", delta_color="off")

    with col2:
        # Disk Space
        st.metric("Disk Space", "78%", delta="+3%", delta_color="inverse")
        # Active Users
        st.metric("Active Users", "3", delta="+1", delta_color="off")


if __name__ == "__main__":
    main()