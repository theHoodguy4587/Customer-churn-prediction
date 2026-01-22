import streamlit as st
import pandas as pd
import joblib

model=joblib.load('models/logistic_regression_model.pkl')
scaler=joblib.load('models/scaler.pkl')
feature_columns=joblib.load('models/feature_names.pkl')
num_cols=joblib.load('models/num_cols_to_scale.pkl')

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



default_features = {
    'Partner_Yes': 1,
    'Dependents_Yes': 0,
    'PhoneService_Yes': 1,
    'MultipleLines_Yes': 0,
    'PaperlessBilling_Yes': 1,
    'OnlineSecurity_Yes': 0,
    'OnlineBackup_Yes': 0,
    'DeviceProtection_Yes': 0,
    'StreamingTV_Yes': 0,
    'StreamingMovies_Yes': 0,
}

for col, val in default_features.items():
    if col in input_encoded.columns:
        input_encoded[col] = val

if internet_service == "No":
    no_internet_cols = [
        'OnlineSecurity_No internet service',
        'OnlineBackup_No internet service',
        'DeviceProtection_No internet service',
        'TechSupport_No internet service',
        'StreamingTV_No internet service',
        'StreamingMovies_No internet service'
    ]
    for col in no_internet_cols:
        if col in input_encoded.columns:
            input_encoded[col] = 1


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



    st.markdown("### Customer Profile Summary")
    st.write({
        "Tenure (months)": tenure,
        "Monthly Charges": monthly_charges,
        "Contract Type": contract_type,
        "Internet Service": internet_service,
        "Payment Method": payment_method,
        "Tech Support": tech_support
    })

    st.progress(min(prediction_proba, 1.0))
    st.caption("Churn Risk Level")




    st.markdown("""
    ### Business Use Case
    This model helps telecom companies identify high-risk customers early, enabling proactive retention strategies such as targeted offers, improved support, or contract adjustments.
    """)


