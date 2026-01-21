import streamlit as st
import pandas as pd
import joblib

model=joblib.load('models/logistic_regression_model.pkl')
scaler=joblib.load('models/scaler.pkl')
feature_columns=joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.write("Predicts whether a customer is likely to churn based on service usage.")

st.sidebar.header("Input Customer Data")

tenure=st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges=st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges=tenure * monthly_charges


contract_type=st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service=st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method=st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support=st.sidebar.selectbox("Tech Support", ["Yes", "No"])

raw_input={
   'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract_type,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'TechSupport': tech_support
}

input_df=pd.DataFrame([raw_input])
input_encoded=pd.get_dummies(input_df)



input_encoded=input_encoded.reindex(columns=feature_columns, fill_value=0)





num_cols=['tenure', 'MonthlyCharges', 'TotalCharges']



input_encoded[num_cols]=scaler.transform(input_encoded[num_cols])

if st.button("Predict Churn"):
    prediction=model.predict(input_encoded)[0]
    prediction_proba=model.predict_proba(input_encoded)[0][1]

    if prediction==1:
        st.error(f"High Risk of churn! (Probability: {prediction_proba:.2f})")
        st.write("Consider offering retention incentives or improving customer support.")
    else:
        st.success(f"The customer is unlikely to churn. (Probability: {prediction_proba:.2f})")
        st.write("Focus on maintaining current service quality to keep the customer satisfied.")