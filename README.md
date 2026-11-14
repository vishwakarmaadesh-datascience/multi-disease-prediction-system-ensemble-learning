# Multi-Disease Prediction System Using Ensemble Learning

An end-to-end Machine Learning project that integrates multiple predictive models into a unified healthcare application to estimate the risk of five major diseases using patient health parameters.

The system uses ensemble learning techniques, data preprocessing, feature engineering and trained classification models to generate predictions through an interactive Streamlit application.

## Project Overview

Traditional disease prediction systems often focus on identifying a single medical condition. This project aims to bring multiple disease prediction models together into one application, allowing users to evaluate different health conditions through a unified interface.

The system uses structured healthcare datasets, applies appropriate preprocessing techniques, trains and evaluates multiple machine learning algorithms, and integrates the trained models into an interactive application.

## Supported Disease Predictions

The application supports the following five disease prediction modules:

| Disease | Prediction Task |
|---|---|
| Diabetes | Predicts diabetes risk using medical and diagnostic parameters |
| Heart Disease | Estimates the likelihood of heart disease |
| Stroke | Estimates stroke risk based on patient health indicators |
| Chronic Kidney Disease | Predicts potential kidney disease |
| Liver Disease | Estimates the likelihood of liver disease |

## Key Features

- **Multi-Disease Prediction:** Five independent disease prediction modules in one application.
- **Ensemble Learning:** Utilizes multiple machine learning algorithms to compare and improve predictive performance.
- **Data Preprocessing:** Handles missing values, inconsistent data, categorical variables and numerical features.
- **Feature Engineering:** Applies feature transformations and appropriate feature preparation techniques.
- **Model Evaluation:** Compares model performance using classification evaluation metrics.
- **Model Persistence:** Saves trained models and preprocessing artifacts using Joblib.
- **Interactive Interface:** Provides a user-friendly Streamlit application for entering patient parameters and viewing predictions.
- **Modular Architecture:** Separates application logic, preprocessing, prediction and model loading.

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. **Data Collection:** Obtaining structured healthcare datasets for five diseases.
2. **Exploratory Data Analysis:** Examining distributions, missing values, correlations and class imbalance.
3. **Data Preprocessing:** Handling missing values, invalid entries, categorical variables and numerical features.
4. **Feature Engineering:** Preparing relevant features for model training.
5. **Model Development:** Training and comparing multiple classification algorithms.
6. **Ensemble Learning:** Combining multiple classifiers using a soft voting approach where applicable.
7. **Model Evaluation:** Evaluating performance using accuracy, precision, recall, F1-score and confusion matrices.
8. **Model Serialization:** Saving trained models and preprocessing objects using Joblib.
9. **Application Development:** Integrating trained models into a Streamlit interface.
10. **Deployment:** Preparing the application for cloud deployment.

## Algorithms Used

The machine learning workflow explores the following classification algorithms:

| Algorithm | Application |
|---|---|
| Logistic Regression | Binary classification |
| Random Forest | Ensemble-based classification |
| Gradient Boosting | Boosted decision-tree classification |
| XGBoost | Gradient boosting framework |
| Support Vector Machine (SVM) | Classification using optimal decision boundaries |

**Ensemble Technique:** Soft Voting Classifier combines the predicted class probabilities of multiple classifiers to determine the final prediction.

The final model configuration and preprocessing pipeline may differ across individual disease datasets.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Data Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Project Structure

```text
Multi-Disease-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── stroke_model.pkl
│   ├── kidney_model.pkl
│   └── liver_model.pkl
│
├── scalers/
│   ├── diabetes_scaler.pkl
│   ├── heart_scaler.pkl
│   ├── stroke_scaler.pkl
│   ├── kidney_scaler.pkl
│   └── liver_scaler.pkl
│
├── config/
│   └── final_features.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── model_loader.py
│   └── utils.py
│
├── notebooks/
│   └── multi_disease_prediction.ipynb
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── healthcare-dataset-stroke-data.csv
│   ├── kidney_disease.csv
│   └── Liver_disease_data.csv
│
└── assets/
    └── images/
```

*Note: The structure represents the intended project organization. Update it if your actual repository differs.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vishwakarmaadesh-datascience/multi-disease-prediction-system-ensemble-learning.git
```

### 2. Navigate to the Project Directory

```bash
cd multi-disease-prediction-system-ensemble-learning
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be accessible at:

```text
http://localhost:8501
```

## Model Evaluation

The models are evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics provide different perspectives on classification performance, particularly when dealing with imbalanced healthcare datasets.

Actual model performance should be reported separately for each disease using the held-out test dataset.

## Potential Future Improvements

- Integrate explainable AI techniques such as SHAP.
- Develop a centralized prediction API using FastAPI.
- Introduce automated model monitoring and versioning using MLflow.
- Explore advanced deep learning architectures.
- Improve model interpretability and uncertainty estimation.
- Implement automated testing and continuous integration.

## Important Medical Disclaimer

This project is developed for educational, academic and research purposes.

The predictions generated by this application are based on machine learning models trained on historical datasets. They are experimental estimates and must not be interpreted as professional medical diagnoses or recommendations.

Users should consult qualified healthcare professionals for medical evaluation and treatment decisions.

## Author

**Adesh Vishwakarma**

MSc Data Science Student | BSc Data Science Graduate | Aspiring Machine Learning Engineer

- **GitHub:** https://github.com/vishwakarmaadesh-datascience
- **LinkedIn:** https://www.linkedin.com/in/adesh-vishwakarma-53108b324
- **Live Demo:** https://multi-disease-prediction-system-ensemble-learning-npulxwhpkffa.streamlit.app/

---

*Developed as an end-to-end Machine Learning project exploring ensemble learning and multi-disease prediction.*
