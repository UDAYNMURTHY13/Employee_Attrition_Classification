
import streamlit as st
import pickle
import pandas as pd
from pyngrok import ngrok

# Load model, scaler, and feature names
@st.cache_resource
def load_model():
    with open('models\best_attrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Streamlit UI
st.title("Employee Attrition Prediction")
st.write("Enter employee details below to predict the likelihood of attrition.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.selectbox("OverTime", options=["Yes", "No"])
job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4])

# Create input dictionary
employee_input = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'YearsAtCompany': years_at_company,
    'OverTime': overtime,
    'JobSatisfaction': job_satisfaction
}

# Convert to DataFrame
input_df = pd.DataFrame([employee_input])

# Preprocessing: One-hot encode categorical variables
input_df_encoded = pd.get_dummies(input_df)

# Align with training features
for col in feature_names:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[feature_names]

# Scale features
input_scaled = scaler.transform(input_df_encoded)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")
    st.write(f"Probability of Attrition: {probability:.2f}")
