import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# ===================================
# ⚙️ Load Model, Scaler, and Features
# ===================================
@st.cache_resource
def load_model():
    with open('models/best_attrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ===================================
# 🖥️ Page Setup
# ===================================
st.set_page_config(page_title="Employee Attrition Prediction", page_icon="💼", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Attrition Predictor")
    st.markdown("### Developed by **Uday Narasimha Murthy**")
    st.markdown("Predict employee attrition likelihood with actionable HR insights.")
    st.markdown("---")
    st.markdown("**Model:** XGBoost with ADASYN balancing")
    st.markdown("**Metrics:** Accuracy, F1, ROC-AUC")
    st.markdown("**Explainability:** SHAP")
    st.markdown("---")
    st.markdown("© 2025 | AI HR Analytics")

# ===================================
# 📋 Input Form
# ===================================
st.title("💼 Employee Attrition Prediction Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
with col2:
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    overtime = st.selectbox("OverTime", options=["Yes", "No"])
with col3:
    job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4])
    work_life_balance = st.selectbox("Work-Life Balance (1=Bad, 4=Excellent)", options=[1, 2, 3, 4])

# ===================================
# 🧮 Data Preparation
# ===================================
employee_input = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'YearsAtCompany': years_at_company,
    'OverTime': overtime,
    'JobSatisfaction': job_satisfaction,
    'WorkLifeBalance': work_life_balance
}
input_df = pd.DataFrame([employee_input])
input_df_encoded = pd.get_dummies(input_df)

for col in feature_names:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[feature_names]
input_scaled = scaler.transform(input_df_encoded)

# ===================================
# 🤖 Prediction Section
# ===================================
if st.button("🔍 Predict Attrition Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Summary")

    if probability < 0.3:
        risk_level = "🟢 Low Risk"
        color = "green"
    elif probability < 0.6:
        risk_level = "🟡 Medium Risk"
        color = "orange"
    else:
        risk_level = "🔴 High Risk"
        color = "red"

    st.write(f"**Prediction:** {'Attrition' if prediction == 1 else 'No Attrition'}")
    st.write(f"**Attrition Probability:** {probability:.2f}")
    st.markdown(f"**Risk Level:** <span style='color:{color}; font-size:22px'><b>{risk_level}</b></span>", unsafe_allow_html=True)
    st.progress(float(probability))

    # ===================================
    # 💬 HR Insights
    # ===================================
    st.markdown("---")
    st.header("💡 Actionable HR Insights")

    insights = []
    if probability >= 0.6:
        insights.append("• High attrition risk detected — prioritize retention strategy.")
    elif probability >= 0.3:
        insights.append("• Moderate risk — maintain engagement and growth discussions.")
    else:
        insights.append("• Low risk — continue strong employee experience.")

    if overtime == "Yes":
        insights.append("• Employee works overtime — assess workload and promote balance.")
    if job_satisfaction <= 2:
        insights.append("• Low job satisfaction — explore causes via 1-on-1 discussion.")
    if years_at_company > 10 and job_satisfaction <= 2:
        insights.append("• Long-term employee showing dissatisfaction — potential burnout risk.")
    if monthly_income < 5000 and age > 35:
        insights.append("• Income below peers — review compensation fairness.")
    if work_life_balance <= 2:
        insights.append("• Poor work-life balance — offer flexibility or time-off options.")

    for insight in insights:
        st.markdown(f"- {insight}")

    # ===================================
    # 🧭 HR Recommendations
    # ===================================
    st.markdown("---")
    st.header("🧭 Recommended HR Actions")
    if probability >= 0.6:
        st.markdown("""
        **Immediate Actions:**
        - Conduct retention interview.
        - Offer skill development or promotion path.
        - Adjust compensation if needed.
        - Strengthen recognition programs.
        """)
    elif probability >= 0.3:
        st.markdown("""
        **Preventive Actions:**
        - Conduct quarterly engagement surveys.
        - Recognize contributions in team meetings.
        - Encourage mentorship and flexible scheduling.
        """)
    else:
        st.markdown("""
        **Sustain Actions:**
        - Continue current engagement and benefits.
        - Offer leadership opportunities and learning growth.
        """)

    # ===================================
# 🔍 SHAP Explainability (Fixed)
# ===================================
st.markdown("---")
st.header("📈 Explainability — SHAP Feature Impact")

# Use TreeExplainer for XGBoost models
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # Create a waterfall or bar plot safely
    st.subheader("Feature Contribution to Attrition Prediction")

    shap_df = pd.DataFrame({
        "Feature": input_df_encoded.columns,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap_df.plot(kind="barh", x="Feature", y="SHAP Value", ax=ax)
    plt.title("Top Feature Impacts on Prediction")
    st.pyplot(fig)

except Exception as e:
    st.error("⚠️ SHAP explainability could not be generated for this model type.")
    st.exception(e)


    # ===================================
    # 🧾 Generate Mini Report
    # ===================================
    st.markdown("---")
    st.header("📄 HR Report Summary")
    st.markdown(f"""
    **Employee Profile Summary:**
    - Age: {age}
    - Years at Company: {years_at_company}
    - Monthly Income: {monthly_income}
    - Overtime: {overtime}
    - Job Satisfaction: {job_satisfaction}
    - Work-Life Balance: {work_life_balance}

    **Prediction:** {'Attrition' if prediction == 1 else 'No Attrition'}  
    **Risk Probability:** {probability:.2f}  
    **Risk Level:** {risk_level}
    """)

    # ===================================
    # 📥 Downloadable Report
    # ===================================
    pdf_text = f"""
    Employee Attrition Prediction Report

    Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}
    Probability: {probability:.2f}
    Risk Level: {risk_level}

    Key Insights:
    {chr(10).join(insights)}

    Recommended HR Actions:
    {'Immediate' if probability >= 0.6 else 'Preventive' if probability >= 0.3 else 'Sustain'} actions suggested.
    """

    buffer = BytesIO()
    buffer.write(pdf_text.encode())
    buffer.seek(0)
    st.download_button("📥 Download Report", data=buffer, file_name="Attrition_Report.txt", mime="text/plain")

    # ===================================
    # 📝 HR Feedback
    # ===================================
    st.markdown("---")
    st.header("🗒️ HR Follow-up Notes")
    hr_notes = st.text_area("Write observations or follow-up actions here:")
    if st.button("💾 Save Notes"):
        st.success("Notes saved locally for this session (not persistent).")
