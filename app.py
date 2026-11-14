import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Input ranges and guidance
ranges = {
    "Diabetes": {
        'Pregnancies': (0, 17),
        'Glucose': (44, 199),
        'BloodPressure': (0, 122),
        'SkinThickness': (7, 99),
        'Insulin': (16, 846),
        'BMI': (18.2, 67.1),
        'DiabetesPedigreeFunction': (0.078, 2.42),
        'Age': (21, 81)
    },
    "Stroke": {
        'age': (0.08, 79),
        'hypertension': (0, 1),
        'heart_disease': (0, 1),
        'avg_glucose_level': (55.1, 271.7),
        'bmi': (10.3, 97.6),
        'gender': ['Female', 'Male', 'Other'],
        'ever_married': ['Yes', 'No'],
        'work_type': ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'],
        'Residence_type': ['Urban', 'Rural'],
        'smoking_status': ['formerly smoked', 'never smoked', 'smokes']
    },
    "Heart": {
        'age': (29, 77),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (71, 202),
        'exang': (0, 1),
        'oldpeak': (0, 6.2),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 3)
    },
    "Kidney": {
        'age': (2, 90),
        'bp': (50, 180),
        'sg': (1.005, 1.025),
        'al': (0, 5),
        'su': (0, 5),
        'rbc': (0, 1),
        'pc': (0, 1),
        'pcc': (0, 1),
        'ba': (0, 1),
        'bgr': (22, 490),
        'bu': (1.5, 391),
        'sc': (0.4, 76),
        'sod': (111, 150),
        'pot': (2.5, 47),
        'hemo': (3.1, 17.8),
        'pcv': (9, 54),
        'wc': (2200, 26400),
        'rc': (2.1, 8.0),
        'htn': (0, 1),
        'dm': (0, 1),
        'cad': (0, 1),
        'appet': (0, 1),
        'pe': (0, 1),
        'ane': (0, 1)
    },
    "Liver": {
        'Age': (1, 90),
        'Gender': ['Male', 'Female'],
        'BMI': (12.4, 47.9),
        'AlcoholConsumption': (0, 71),
        'Smoking': (0, 1),
        'GeneticRisk': (0, 1),
        'PhysicalActivity': (0, 10),
        'Diabetes': (0, 1),
        'Hypertension': (0, 1),
        'LiverFunctionTest': (0, 100)
    }
}

original_features_for_input = {
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "Stroke": ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'],
    "Heart": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    "Kidney": ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
    "Liver": ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest']
}

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "notebooks" / "final_features.pkl"

final_features = joblib.load(FEATURES_PATH)

# Load models and scalers
@st.cache_resource
def load_model_and_scaler(disease):
    model_path = f"models/{disease.lower()}_model.pkl"
    scaler_path = f"scalers/{disease.lower()}_scaler.pkl"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Preprocessing functions
def preprocess_diabetes(df):
    df = df.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    for col in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def preprocess_stroke(df):
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def preprocess_heart(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def preprocess_kidney(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    return df

def preprocess_liver(df):
    df = df.copy()
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    return df

def _apply_categorical_encoding_consistent(df, reference_columns):
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        if df_processed[col].nunique() == 2:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    df_processed = df_processed.reindex(columns=reference_columns, fill_value=0)
    return df_processed

def prepare_new_data_for_prediction(new_data_dict, selected_disease, final_feature_columns):
    df_raw_input = pd.DataFrame([new_data_dict])
    if selected_disease == "Diabetes":
        df_processed_temp = preprocess_diabetes(df_raw_input)
    elif selected_disease == "Stroke":
        df_processed_temp = preprocess_stroke(df_raw_input)
    elif selected_disease == "Heart":
        df_processed_temp = preprocess_heart(df_raw_input)
    elif selected_disease == "Kidney":
        df_processed_temp = preprocess_kidney(df_raw_input)
    elif selected_disease == "Liver":
        df_processed_temp = preprocess_liver(df_raw_input)
    else:
        df_processed_temp = df_raw_input
    df_processed = _apply_categorical_encoding_consistent(df_processed_temp, final_feature_columns)
    return df_processed

# ──────────────────────────────────────────────
#               Streamlit App
# ──────────────────────────────────────────────

st.title("Multi-Disease Prediction App")

st.markdown("""
Select a disease and enter patient details to get a risk prediction.
**This is for educational purposes only — always consult a doctor.**
""")

diseases = list(ranges.keys())
selected_disease = st.selectbox("Choose Disease", diseases)

if selected_disease:
    try:
        model, scaler = load_model_and_scaler(selected_disease)
    except Exception as e:
        st.error(f"Could not load model/scaler for {selected_disease}. Check training files.")
        st.stop()

    st.subheader(f"Enter Details for {selected_disease}")

    new_data = {}
    for feature in original_features_for_input[selected_disease]:
        if feature in ranges[selected_disease]:
            range_info = ranges[selected_disease][feature]
            label = feature.replace("_", " ").capitalize()

            if isinstance(range_info, list):  # Categorical
                new_data[feature] = st.selectbox(
                    label,
                    range_info,
                    help="Choose from the available options"
                )
            else:  # Numeric
                min_val, max_val = range_info
                new_data[feature] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1 if (max_val - min_val) < 10 else 1.0,
                    help=f"Typical range: {min_val} — {max_val}"
                )

    if st.button("Predict Risk", type="primary"):
        with st.spinner("Analyzing..."):
            prepared_df = prepare_new_data_for_prediction(new_data, selected_disease, final_features[selected_disease])
            scaled_input = scaler.transform(prepared_df)
            pred = model.predict(scaled_input)[0]
            probs = model.predict_proba(scaled_input)[0]
            prob_positive = probs[1] if len(probs) > 1 else probs[0]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"**High Risk Detected** — Probability: {prob_positive:.1%}")
        else:
            st.success(f"**Low Risk** — Probability of disease: {prob_positive:.1%}")

        st.info("This is an ML model estimate only. Please consult a healthcare professional for accurate diagnosis.")
