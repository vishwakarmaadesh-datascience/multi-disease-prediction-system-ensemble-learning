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
* **Multi-Disease Support:** Predicts [Insert Diseases, e.g., Diabetes, Heart Disease] within a single app.
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
## Streamlit Prediction
1. For Diabetes
<img width="466" height="759" alt="image" src="https://github.com/user-attachments/assets/67a93cfe-47f7-48ef-9399-8942c6d3e3f1" />
2. For Stroke
<img width="464" height="862" alt="image" src="https://github.com/user-attachments/assets/67783d9c-75fd-4beb-a656-00f66fd64182" />
3. For Heart
<img width="314" height="689" alt="image" src="https://github.com/user-attachments/assets/ca8d66c3-a9f8-4c0e-9d83-93630f6578f6" />
4. For Liver
<img width="465" height="894" alt="image" src="https://github.com/user-attachments/assets/38e5639e-1bfe-42cd-8aa7-62fc665b43ad" />



Adesh Vishwakarma 
vishwakarmaadesh90@gmail.com
