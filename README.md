# Multi-Disease Prediction System

A robust machine learning application that predicts multiple health conditions (e.g., Diabetes, Heart Disease, and Kidney Disease) using ensemble learning techniques. This project features a user-friendly web interface built with Streamlit.

##  Overview
Early diagnosis is critical in healthcare. This project leverages historical patient data to provide instant predictions. By using ensemble methods like **Random Forest** and **XGBoost**, the system achieves higher accuracy and reliability compared to single-model approaches.

##  Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
* **Web Framework:** Streamlit
* **Environment:** Google Colab / Local IDE

##  Key Features
* **Multi-Disease Support:** Predicts [Insert Diseases, e.g., Diabetes, Heart Disease, Parkinson's] within a single app.
* **Ensemble Learning:** Utilizes advanced algorithms to minimize false negatives.
* **Interactive UI:** Simple sliders and input fields for medical parameters.
* **Data Preprocessing:** Includes handling missing values, feature scaling, and categorical encoding.

##  Project Structure
* `Multi_Disease_Prediction.ipynb`: The core logic, EDA, and model training.
* `app.py`: The Streamlit web application script.
* `models/`: Pre-trained serialized models (.pkl files).
* `requirements.txt`: List of dependencies for easy installation.

##  How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
2. Install Dependencies
 pip install -r requirements.txt
3.Launch app
streamlit run app.py 
