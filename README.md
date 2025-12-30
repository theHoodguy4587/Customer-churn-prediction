# 📉 Customer Churn Prediction – End-to-End Machine Learning Project

## 📌 Project Overview
Customer churn is a critical business problem where companies aim to identify customers who are likely to stop using their services.  
This project builds an **end-to-end machine learning pipeline** to predict customer churn using historical customer data.

The project simulates a **real-world data science workflow**, covering:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Model and scaler persistence for deployment readiness

---

## 🎯 Business Objective
- Predict whether a customer will churn (`Yes` / `No`)
- Identify **key factors driving churn**
- Help businesses take **proactive retention actions**

---

## 🗂️ Project Structure
Customer-churn-prediction/
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Modeling_and_Evaluation.ipynb
│
├── models/
│ ├── scaler.pkl
│ ├── logistic_model.pkl
│ ├── random_forest_model.pkl
│ └── feature_names.pkl
│
├── data/
│ └── customer_churn.csv
| └── X_test_scaled.npy
| └── X_train_scaled.npy
| └── y_test.csv
| └── y_train.csv
│
├── README.md
└── requirements.txt



---

## 📊 Dataset Description
The dataset contains customer-level information such as:
- Demographics (gender, senior citizen status)
- Service usage (internet service, contract type)
- Billing information (monthly charges, total charges)
- Target variable: **Churn**

### Target Variable
- `Churn`  
  - 1 → Customer churned  
  - 0 → Customer retained

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- **Short-tenure** customers show higher churn probability.
- **Higher monthly charges** correlate with churn.
- **Month-to-month** contracts are the strongest churn indicator.
- **Electronic-check payement** methods show higher churn risk.
EDA includes:
- Univariate and bivariate analysis
- Distribution plots
- Churn comparison across categorical variables

---

## 🛠️ Feature Engineering
Performed in `02_Feature_Engineering.ipynb`

Steps:
- Converted `TotalCharges` to numeric
- Handled missing values using **median imputation**
- One-hot encoded categorical variables (`drop_first=True`)
- Saved feature names for future model interpretation
- Applied **StandardScaler** to numerical features
- Prevented data leakage by fitting scaler only on training data

---

## 🤖 Model Building
Implemented in `03_Modeling_and_Evaluation.ipynb`

### Models Used
- **Logistic Regression**
- **Random Forest Classifier**

### Why These Models?
- Logistic Regression: baseline, interpretable
- Random Forest: captures non-linear relationships and feature interactions

---

## 📈 Model Evaluation
Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Curve

### Key Results
- Logistic Regression outperformed  Random Forest on ROC-AUC
- Logistic Regression provided strong interpretability
- ROC curves used to compare classifier performance

---

## 🔑 Feature Importance
- Logistic Regression coefficients analyzed
- Random Forest feature importance extracted
- Identified key churn drivers such as:
  - Contract type
  - Tenure
  - Monthly charges
  - Payment method

---

## 💾 Model Persistence
Saved for deployment readiness:
- Trained models (`.pkl`)
- Scaler object
- Feature names

This allows:
- Reuse without retraining
- Integration with web apps (Streamlit / Flask)
- Future inference on new data

---

## 🚀 Tools & Technologies
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **Joblib**
- **Jupyter Notebook**

---

## 📌 Key Takeaways
- End-to-end ML project following industry best practices
- Clean separation of EDA, feature engineering, and modeling
- Production-ready pipeline with saved artifacts
- Strong business relevance and interpretability

---

## 🔮 Future Improvements
- Hyperparameter tuning
- Cross-validation
- XGBoost / LightGBM models
- Model deployment using Streamlit
- Monitoring model drift

---

## 👤 Author
**Senitha Gunathilaka**  
Aspiring Data Scientist  
📌 LinkedIn: *(https://www.linkedin.com/in/senitha-gunathilaka-404236285/)*  

---

