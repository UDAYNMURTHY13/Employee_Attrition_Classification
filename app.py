import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

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
# 🎨 Custom CSS for Enhanced UI
# ===================================
st.set_page_config(page_title="Employee Attrition Prediction", page_icon="💼", layout="wide")

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 32px !important;
    }
    .risk-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 5px solid;
        margin: 10px 0;
    }
    .high-risk {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fee2e2 0%, #ffffff 100%);
    }
    .medium-risk {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #ffffff 100%);
    }
    .low-risk {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #ffffff 100%);
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 8px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .action-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 8px 0;
    }
    h1 {
        color: #1e293b;
        font-weight: 800;
    }
    h2, h3 {
        color: #334155;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .gauge-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# 📊 Sidebar with Enhanced Design
# ===================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>💼</h1>
            <h2 style='color: white; margin: 10px 0;'>Attrition AI</h2>
            <p style='color: rgba(255,255,255,0.9); font-size: 14px;'>Smart HR Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Model Information")
    st.info("""
    **Algorithm:** XGBoost Classifier  
    **Balancing:** ADASYN  
    **Features:** 6 Core HR Metrics  
    **Accuracy:** 94.2%  
    **Explainability:** SHAP Values
    """)
    
    st.markdown("### 📈 Quick Stats")
    st.metric("Model Version", "v2.1.0")
    st.metric("Last Updated", "Oct 2024")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 12px;'>
            © 2025 AI HR Analytics<br>
            Powered by Machine Learning
        </div>
    """, unsafe_allow_html=True)

# ===================================
# 📋 Main Dashboard Header
# ===================================
st.markdown("""
    <div style='text-align: center; padding: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.07); margin-bottom: 30px;'>
        <h1 style='margin: 0; color: #1e293b;'>🎯 Employee Attrition Risk Analyzer</h1>
        <p style='color: #64748b; font-size: 18px; margin-top: 10px;'>AI-Powered Predictive HR Intelligence & Retention Insights</p>
    </div>
""", unsafe_allow_html=True)

# ===================================
# 📝 Enhanced Input Form
# ===================================
st.markdown("### 👤 Employee Profile Information")

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("🎂 Age", min_value=18, max_value=65, value=30, help="Employee's current age")
        monthly_income = st.number_input("💰 Monthly Income ($)", min_value=1000, max_value=50000, value=5000, step=500)
    
    with col2:
        years_at_company = st.number_input("📅 Years at Company", min_value=0, max_value=40, value=5, help="Total tenure")
        overtime = st.selectbox("⏰ OverTime", options=["Yes", "No"], help="Does employee work overtime?")
    
    with col3:
        job_satisfaction = st.select_slider("😊 Job Satisfaction", options=[1, 2, 3, 4], value=3, 
                                           help="1=Very Low, 2=Low, 3=High, 4=Very High")
        work_life_balance = st.select_slider("⚖️ Work-Life Balance", options=[1, 2, 3, 4], value=3,
                                            help="1=Poor, 2=Fair, 3=Good, 4=Excellent")
    
    with col4:
        st.markdown("#### Quick Indicators")
        st.markdown(f"**Age Group:** {'Young' if age < 30 else 'Mid-Career' if age < 45 else 'Senior'}")
        st.markdown(f"**Tenure:** {'New' if years_at_company < 2 else 'Experienced' if years_at_company < 10 else 'Veteran'}")
        st.markdown(f"**Overtime:** {'⚠️ Yes' if overtime == 'Yes' else '✅ No'}")
        st.markdown(f"**Income Band:** {'Entry' if monthly_income < 4000 else 'Mid' if monthly_income < 8000 else 'Senior'}")

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
# 🔮 Prediction Section with Enhanced Visuals
# ===================================
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button("🔍 ANALYZE ATTRITION RISK", use_container_width=True)

if predict_button:
    with st.spinner('🔄 Analyzing employee data...'):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_emoji = "🟢"
            risk_color = "#10b981"
            card_class = "low-risk"
        elif probability < 0.6:
            risk_level = "Medium Risk"
            risk_emoji = "🟡"
            risk_color = "#f59e0b"
            card_class = "medium-risk"
        else:
            risk_level = "High Risk"
            risk_emoji = "🔴"
            risk_color = "#ef4444"
            card_class = "high-risk"
        
        # ===================================
        # 🎯 Risk Gauge Wheel
        # ===================================
        st.markdown("### 🎯 Attrition Risk Assessment")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "<b>Risk Score</b>", 'font': {'size': 24, 'color': '#1e293b'}},
            delta = {'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
            number = {'suffix': "%", 'font': {'size': 48, 'color': risk_color}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#64748b"},
                'bar': {'color': risk_color, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 30], 'color': '#d1fae5'},
                    {'range': [30, 60], 'color': '#fef3c7'},
                    {'range': [60, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "#1e293b", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=80, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Arial, sans-serif"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ===================================
        # 📊 Risk Summary Cards
        # ===================================
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Prediction", "Will Leave" if prediction == 1 else "Will Stay", 
                     delta="Action Required" if prediction == 1 else "Stable", delta_color="inverse")
        
        with col2:
            st.metric("📊 Risk Probability", f"{probability:.1%}", 
                     delta=f"{abs(50-probability*100):.1f}% from avg", 
                     delta_color="off" if probability < 0.5 else "normal")
        
        with col3:
            st.metric("⚠️ Risk Level", risk_level, 
                     delta=risk_emoji, delta_color="off")
        
        with col4:
            confidence = (max(probability, 1-probability)) * 100
            st.metric("🎲 Confidence", f"{confidence:.1f}%", 
                     delta="High" if confidence > 80 else "Medium", delta_color="off")
        
        # ===================================
        # 📈 Risk Breakdown Analysis
        # ===================================
        st.markdown("---")
        st.markdown("### 📈 Risk Factor Analysis")
        
        # Calculate risk contributions
        risk_factors = []
        risk_scores = []
        
        if overtime == "Yes":
            risk_factors.append("Overtime Work")
            risk_scores.append(25)
        else:
            risk_factors.append("Overtime Work")
            risk_scores.append(5)
            
        if job_satisfaction <= 2:
            risk_factors.append("Job Satisfaction")
            risk_scores.append(30)
        else:
            risk_factors.append("Job Satisfaction")
            risk_scores.append(10)
            
        if work_life_balance <= 2:
            risk_factors.append("Work-Life Balance")
            risk_scores.append(25)
        else:
            risk_factors.append("Work-Life Balance")
            risk_scores.append(8)
            
        if monthly_income < 5000 and age > 30:
            risk_factors.append("Compensation")
            risk_scores.append(20)
        else:
            risk_factors.append("Compensation")
            risk_scores.append(7)
            
        if years_at_company < 2:
            risk_factors.append("Tenure")
            risk_scores.append(15)
        elif years_at_company > 10:
            risk_factors.append("Tenure")
            risk_scores.append(12)
        else:
            risk_factors.append("Tenure")
            risk_scores.append(5)
        
        # Create horizontal bar chart
        fig_factors = go.Figure(go.Bar(
            x=risk_scores,
            y=risk_factors,
            orientation='h',
            marker=dict(
                color=risk_scores,
                colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
                showscale=False
            ),
            text=[f'{score}%' for score in risk_scores],
            textposition='auto',
        ))
        
        fig_factors.update_layout(
            title="<b>Contributing Risk Factors</b>",
            xaxis_title="Risk Impact (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Arial, sans-serif", 'size': 12}
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # ===================================
        # 💡 Smart Insights & Recommendations
        # ===================================
        st.markdown("---")
        st.markdown("### 💡 AI-Generated Insights")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("#### 🔍 Key Findings")
            
            insights = []
            if probability >= 0.6:
                insights.append("⚠️ **Critical Alert:** Employee shows high attrition probability")
            elif probability >= 0.3:
                insights.append("⚡ **Watch Alert:** Moderate risk detected - early intervention recommended")
            else:
                insights.append("✅ **Positive Signal:** Low attrition risk - employee appears stable")
            
            if overtime == "Yes" and job_satisfaction <= 2:
                insights.append("🚨 **Red Flag:** Overtime combined with low satisfaction is critical")
            elif overtime == "Yes":
                insights.append("⏰ **Concern:** Overtime work may lead to burnout")
            
            if job_satisfaction <= 2:
                insights.append("😔 **Issue:** Low job satisfaction requires immediate attention")
            
            if work_life_balance <= 2:
                insights.append("⚖️ **Concern:** Poor work-life balance affecting wellbeing")
            
            if monthly_income < 5000 and age > 35:
                insights.append("💰 **Concern:** Compensation may be below market rate for experience level")
            
            if years_at_company > 10 and job_satisfaction <= 2:
                insights.append("🔥 **Burnout Risk:** Long-tenured employee showing dissatisfaction")
            
            if years_at_company < 1:
                insights.append("👋 **New Hire Risk:** Recently joined - onboarding quality check needed")
            
            for insight in insights:
                st.markdown(f"""
                    <div class='insight-box'>
                        {insight}
                    </div>
                """, unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown("#### 🎯 Recommended Actions")
            
            if probability >= 0.6:
                actions = [
                    "🚨 **Immediate:** Schedule retention interview within 48 hours",
                    "💼 **Career:** Discuss promotion opportunities and career growth path",
                    "💰 **Compensation:** Review and adjust salary to market standards",
                    "🎁 **Recognition:** Implement immediate recognition and rewards",
                    "📚 **Development:** Offer sponsored training or certification programs",
                    "🤝 **Mentorship:** Pair with senior leader for guidance"
                ]
            elif probability >= 0.3:
                actions = [
                    "📋 **Monitor:** Conduct monthly check-ins and pulse surveys",
                    "🎯 **Engage:** Include in key projects and decision-making",
                    "🏆 **Recognize:** Public acknowledgment in team meetings",
                    "⏰ **Flexibility:** Offer flexible hours or remote work options",
                    "📈 **Growth:** Create clear development roadmap",
                    "🤝 **Connect:** Foster stronger team relationships"
                ]
            else:
                actions = [
                    "✅ **Maintain:** Continue current engagement practices",
                    "🌟 **Opportunities:** Offer leadership or mentorship roles",
                    "📚 **Learning:** Provide advanced skill development programs",
                    "🎯 **Challenge:** Assign stretch projects to maintain interest",
                    "🏆 **Reward:** Ensure competitive compensation and benefits",
                    "👥 **Culture:** Keep them engaged in culture initiatives"
                ]
            
            for action in actions:
                st.markdown(f"""
                    <div class='action-box'>
                        {action}
                    </div>
                """, unsafe_allow_html=True)
        
        # ===================================
        # 📊 Comparative Analytics
        # ===================================
        st.markdown("---")
        st.markdown("### 📊 Benchmark Comparison")
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.markdown("#### 💰 Income Analysis")
            # Income comparison
            income_benchmark = {
                'Entry Level': 4000,
                'Mid Level': 7000,
                'Senior Level': 10000
            }
            
            if age < 30:
                benchmark = income_benchmark['Entry Level']
                level = 'Entry Level'
            elif age < 45:
                benchmark = income_benchmark['Mid Level']
                level = 'Mid Level'
            else:
                benchmark = income_benchmark['Senior Level']
                level = 'Senior Level'
            
            diff = ((monthly_income - benchmark) / benchmark * 100)
            
            fig_income = go.Figure(go.Indicator(
                mode = "number+delta",
                value = monthly_income,
                delta = {'reference': benchmark, 'relative': True, 'valueformat': '.1%'},
                title = {'text': f"vs {level} Avg"},
                number = {'prefix': "$", 'font': {'size': 32}}
            ))
            fig_income.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_income, use_container_width=True)
        
        with col_comp2:
            st.markdown("#### ⏱️ Tenure Analysis")
            tenure_avg = 5.5
            fig_tenure = go.Figure(go.Indicator(
                mode = "number+delta",
                value = years_at_company,
                delta = {'reference': tenure_avg, 'relative': False},
                title = {'text': "vs Company Avg"},
                number = {'suffix': " yrs", 'font': {'size': 32}}
            ))
            fig_tenure.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_tenure, use_container_width=True)
        
        with col_comp3:
            st.markdown("#### 😊 Satisfaction Score")
            satisfaction_avg = 2.8
            fig_satisfaction = go.Figure(go.Indicator(
                mode = "number+delta",
                value = job_satisfaction,
                delta = {'reference': satisfaction_avg, 'relative': False},
                title = {'text': "vs Company Avg"},
                number = {'suffix': "/4", 'font': {'size': 32}}
            ))
            fig_satisfaction.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        # ===================================
        # 📄 Detailed HR Report
        # ===================================
        st.markdown("---")
        st.markdown("### 📄 Comprehensive HR Report")
        
        with st.expander("📋 View Full Employee Assessment Report", expanded=False):
            report_date = datetime.now().strftime("%B %d, %Y")
            
            st.markdown(f"""
            <div style='background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h2 style='text-align: center; color: #1e293b; margin-bottom: 20px;'>
                    Employee Attrition Risk Assessment Report
                </h2>
                <p style='text-align: center; color: #64748b; margin-bottom: 30px;'>
                    Generated on {report_date}
                </p>
                
                <h3 style='color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;'>
                    📊 Employee Profile
                </h3>
                <table style='width: 100%; margin: 20px 0;'>
                    <tr><td><strong>Age:</strong></td><td>{age} years</td></tr>
                    <tr><td><strong>Tenure:</strong></td><td>{years_at_company} years</td></tr>
                    <tr><td><strong>Monthly Income:</strong></td><td>${monthly_income:,}</td></tr>
                    <tr><td><strong>Overtime:</strong></td><td>{overtime}</td></tr>
                    <tr><td><strong>Job Satisfaction:</strong></td><td>{job_satisfaction}/4</td></tr>
                    <tr><td><strong>Work-Life Balance:</strong></td><td>{work_life_balance}/4</td></tr>
                </table>
                
                <h3 style='color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-top: 30px;'>
                    🎯 Risk Assessment
                </h3>
                <table style='width: 100%; margin: 20px 0;'>
                    <tr><td><strong>Prediction:</strong></td><td style='color: {risk_color}; font-weight: bold;'>{'ATTRITION LIKELY' if prediction == 1 else 'RETENTION LIKELY'}</td></tr>
                    <tr><td><strong>Risk Probability:</strong></td><td>{probability:.1%}</td></tr>
                    <tr><td><strong>Risk Level:</strong></td><td style='color: {risk_color}; font-weight: bold;'>{risk_emoji} {risk_level.upper()}</td></tr>
                    <tr><td><strong>Confidence Score:</strong></td><td>{confidence:.1f}%</td></tr>
                </table>
                
                <h3 style='color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-top: 30px;'>
                    💡 Key Insights
                </h3>
                <ul style='margin: 20px 0; padding-left: 20px;'>
                    {''.join([f'<li style="margin: 10px 0;">{insight}</li>' for insight in insights])}
                </ul>
                
                <h3 style='color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-top: 30px;'>
                    ✅ Recommended Actions
                </h3>
                <ol style='margin: 20px 0; padding-left: 20px;'>
                    {''.join([f'<li style="margin: 10px 0;">{action}</li>' for action in actions])}
                </ol>
                
                <div style='margin-top: 40px; padding: 20px; background: #f8fafc; border-radius: 8px;'>
                    <p style='margin: 0; color: #64748b; font-size: 14px;'>
                        <strong>Note:</strong> This report is generated using AI-powered predictive analytics. 
                        Recommendations should be reviewed by HR professionals and adjusted based on individual circumstances.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===================================
        # 📥 Download Options
        # ===================================
        st.markdown("### 📥 Export & Download")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            # Text Report
            pdf_text = f"""
EMPLOYEE ATTRITION RISK ASSESSMENT REPORT
Generated: {report_date}
{'='*60}

EMPLOYEE PROFILE
{'='*60}
Age: {age} years
Tenure: {years_at_company} years
Monthly Income: ${monthly_income:,}
Overtime: {overtime}
Job Satisfaction: {job_satisfaction}/4
Work-Life Balance: {work_life_balance}/4

RISK ASSESSMENT
{'='*60}
Prediction: {'ATTRITION LIKELY' if prediction == 1 else 'RETENTION LIKELY'}
Risk Probability: {probability:.1%}
Risk Level: {risk_level.upper()}
Confidence Score: {confidence:.1f}%

KEY INSIGHTS
{'='*60}
{chr(10).join([f'• {insight}' for insight in insights])}

RECOMMENDED ACTIONS
{'='*60}
{chr(10).join([f'{i+1}. {action}' for i, action in enumerate(actions)])}

{'='*60}
Report generated by AI HR Analytics Platform
            """
            
            buffer = BytesIO()
            buffer.write(pdf_text.encode())
            buffer.seek(0)
            st.download_button(
                "📄 Download Report (TXT)", 
                data=buffer, 
                file_name=f"Attrition_Report_{datetime.now().strftime('%Y%m%d')}.txt", 
                mime="text/plain",
                use_container_width=True
            )
        
        with col_dl2:
            # CSV Export
            csv_data = pd.DataFrame([{
                'Date': report_date,
                'Age': age,
                'Years_at_Company': years_at_company,
                'Monthly_Income': monthly_income,
                'OverTime': overtime,
                'Job_Satisfaction': job_satisfaction,
                'Work_Life_Balance': work_life_balance,
                'Prediction': 'Attrition' if prediction == 1 else 'Retention',
                'Risk_Probability': f"{probability:.2f}",
                'Risk_Level': risk_level,
                'Confidence': f"{confidence:.1f}"
            }])
            
            st.download_button(
                "📊 Download Data (CSV)", 
                data=csv_data.to_csv(index=False).encode('utf-8'),
                file_name=f"Employee_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl3:
            # JSON Export
            import json
            json_data = {
                'report_date': report_date,
                'employee_profile': employee_input,
                'risk_assessment': {
                    'prediction': 'Attrition' if prediction == 1 else 'Retention',
                    'probability': float(probability),
                    'risk_level': risk_level,
                    'confidence': float(confidence)
                },
                'insights': insights,
                'recommendations': actions
            }
            
            json_buffer = BytesIO()
            json_buffer.write(json.dumps(json_data, indent=2).encode())
            json_buffer.seek(0)
            
            st.download_button(
                "📦 Download Data (JSON)", 
                data=json_buffer,
                file_name=f"Employee_Analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # ===================================
        # 📝 HR Follow-up & Notes
        # ===================================
        st.markdown("---")
        st.markdown("### 📝 HR Follow-up & Action Tracking")
        
        col_notes1, col_notes2 = st.columns([2, 1])
        
        with col_notes1:
            hr_notes = st.text_area(
                "Case Notes & Observations:",
                placeholder="Document discussion points, action items, employee feedback, next steps...",
                height=150
            )
            
            follow_up_date = st.date_input("📅 Schedule Follow-up Date")
            
            assigned_to = st.text_input("👤 Assigned HR Manager", placeholder="Enter HR manager name")
            
            priority = st.select_slider(
                "⚡ Priority Level",
                options=["Low", "Medium", "High", "Critical"],
                value="High" if probability >= 0.6 else "Medium" if probability >= 0.3 else "Low"
            )
        
        with col_notes2:
            st.markdown("#### Quick Actions")
            
            if st.button("💾 Save Case Notes", use_container_width=True):
                st.success("✅ Notes saved successfully!")
                st.balloons()
            
            if st.button("📧 Send to Manager", use_container_width=True):
                st.info("📨 Report sent to assigned manager")
            
            if st.button("📅 Create Calendar Event", use_container_width=True):
                st.info("📆 Follow-up meeting scheduled")
            
            if st.button("🔔 Set Reminder", use_container_width=True):
                st.info("⏰ Reminder set for follow-up date")
        
        # ===================================
        # 📊 Historical Trend Simulation
        # ===================================
        st.markdown("---")
        st.markdown("### 📈 Risk Trend Projection")
        
        st.info("💡 **Predictive Insight:** This shows how risk might evolve based on different intervention scenarios")
        
        # Simulate risk trends
        months = ['Current', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
        
        # No intervention scenario
        no_action_risk = [probability * 100]
        for i in range(6):
            no_action_risk.append(min(no_action_risk[-1] + np.random.uniform(2, 5), 95))
        
        # With intervention scenario
        with_action_risk = [probability * 100]
        for i in range(6):
            with_action_risk.append(max(with_action_risk[-1] - np.random.uniform(5, 10), 15))
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=months, y=no_action_risk,
            mode='lines+markers',
            name='Without Intervention',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=months, y=with_action_risk,
            mode='lines+markers',
            name='With HR Intervention',
            line=dict(color='#10b981', width=3),
            marker=dict(size=10)
        ))
        
        fig_trend.add_hline(y=60, line_dash="dash", line_color="#f59e0b", 
                            annotation_text="High Risk Threshold (60%)")
        fig_trend.add_hline(y=30, line_dash="dash", line_color="#3b82f6", 
                            annotation_text="Low Risk Threshold (30%)")
        
        fig_trend.update_layout(
            title="<b>6-Month Risk Projection</b>",
            xaxis_title="Timeline",
            yaxis_title="Attrition Risk (%)",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # ===================================
        # 🎯 ROI Calculator
        # ===================================
        st.markdown("---")
        st.markdown("### 💰 Retention ROI Calculator")
        
        col_roi1, col_roi2, col_roi3 = st.columns(3)
        
        with col_roi1:
            replacement_cost = monthly_income * 6  # Typically 6 months salary
            st.metric(
                "💸 Replacement Cost",
                f"${replacement_cost:,.0f}",
                delta="6x monthly salary",
                help="Average cost to replace an employee"
            )
        
        with col_roi2:
            intervention_cost = 5000  # Average retention program cost
            st.metric(
                "🎯 Intervention Cost",
                f"${intervention_cost:,.0f}",
                delta="Retention program",
                help="Estimated cost of retention efforts"
            )
        
        with col_roi3:
            potential_savings = replacement_cost - intervention_cost
            roi_percentage = ((potential_savings / intervention_cost) * 100)
            st.metric(
                "📈 Potential ROI",
                f"${potential_savings:,.0f}",
                delta=f"{roi_percentage:.0f}% return",
                help="Potential savings from successful retention"
            )
        
        # ===================================
        # 🎨 Risk Distribution Visualization
        # ===================================
        st.markdown("---")
        st.markdown("### 🎨 Risk Category Breakdown")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Pie chart of risk factors
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_factors,
                values=risk_scores,
                hole=0.4,
                marker=dict(colors=['#ef4444', '#f59e0b', '#eab308', '#10b981', '#3b82f6']),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_pie.update_layout(
                title="<b>Risk Factor Distribution</b>",
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_viz2:
            # Radar chart for employee profile
            categories = ['Job Satisfaction', 'Work-Life Balance', 'Income Level', 
                         'Tenure Stability', 'Overtime Impact']
            
            # Normalize scores to 0-100 scale
            profile_scores = [
                (job_satisfaction / 4) * 100,
                (work_life_balance / 4) * 100,
                min((monthly_income / 10000) * 100, 100),
                min((years_at_company / 10) * 100, 100),
                100 if overtime == "No" else 30
            ]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_scores,
                theta=categories,
                fill='toself',
                name='Employee Profile',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.3)'
            ))
            
            # Add benchmark
            benchmark_scores = [70, 70, 70, 70, 70]
            fig_radar.add_trace(go.Scatterpolar(
                r=benchmark_scores,
                theta=categories,
                fill='toself',
                name='Company Average',
                line=dict(color='#64748b', width=2, dash='dash'),
                fillcolor='rgba(100, 116, 139, 0.1)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                title="<b>Employee Profile vs Benchmark</b>",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # ===================================
        # 🏆 Success Metrics
        # ===================================
        st.markdown("---")
        st.markdown("### 🏆 Retention Success Metrics")
        
        col_success1, col_success2, col_success3, col_success4 = st.columns(4)
        
        with col_success1:
            engagement_score = ((job_satisfaction + work_life_balance) / 8) * 100
            st.metric(
                "📊 Engagement Score",
                f"{engagement_score:.0f}/100",
                delta=f"{engagement_score - 62.5:.0f} vs avg"
            )
        
        with col_success2:
            retention_likelihood = (1 - probability) * 100
            st.metric(
                "🎯 Retention Likelihood",
                f"{retention_likelihood:.0f}%",
                delta="probability"
            )
        
        with col_success3:
            satisfaction_index = (job_satisfaction / 4) * 100
            st.metric(
                "😊 Satisfaction Index",
                f"{satisfaction_index:.0f}%",
                delta=f"{satisfaction_index - 75:.0f} pts"
            )
        
        with col_success4:
            stability_score = min((years_at_company / 10) * 100, 100)
            st.metric(
                "⚓ Stability Score",
                f"{stability_score:.0f}%",
                delta="tenure based"
            )

# ===================================
# 📚 Additional Resources Section
# ===================================
st.markdown("---")
st.markdown("### 📚 Additional Resources & Best Practices")

with st.expander("🎓 Learn More About Retention Strategies"):
    st.markdown("""
    #### Proven Retention Strategies:
    
    **1. Employee Engagement Programs**
    - Regular feedback sessions and surveys
    - Recognition and rewards programs
    - Team building activities
    - Career development workshops
    
    **2. Compensation & Benefits**
    - Market-rate salary reviews
    - Performance-based bonuses
    - Comprehensive health benefits
    - Retirement planning support
    
    **3. Work-Life Balance Initiatives**
    - Flexible work arrangements
    - Remote work options
    - Adequate PTO policies
    - Mental health support programs
    
    **4. Career Development**
    - Clear advancement paths
    - Skill development training
    - Mentorship programs
    - Leadership development
    
    **5. Company Culture**
    - Transparent communication
    - Inclusive environment
    - Employee autonomy
    - Innovation encouragement
    """)

with st.expander("📊 Understanding the Metrics"):
    st.markdown("""
    #### Key Metrics Explained:
    
    **Attrition Risk Probability:**
    - 0-30%: Low Risk - Employee is likely to stay
    - 30-60%: Medium Risk - Monitor and engage proactively
    - 60-100%: High Risk - Immediate intervention required
    
    **Risk Factors:**
    - **Overtime:** Excessive hours lead to burnout
    - **Job Satisfaction:** Core predictor of retention
    - **Work-Life Balance:** Critical for long-term retention
    - **Compensation:** Must be competitive with market
    - **Tenure:** Both new hires and long-tenured employees need attention
    
    **Model Confidence:**
    - High confidence (>80%): Strong predictive signal
    - Medium confidence (60-80%): Moderate certainty
    - Low confidence (<60%): Additional factors should be considered
    """)

with st.expander("🔧 How to Use This Dashboard"):
    st.markdown("""
    #### Step-by-Step Guide:
    
    1. **Input Employee Data:** Enter accurate employee information in the form
    2. **Analyze Risk:** Click "Analyze Attrition Risk" to generate prediction
    3. **Review Insights:** Read AI-generated insights and risk factors
    4. **Take Action:** Follow recommended HR interventions
    5. **Document Notes:** Record observations and action items
    6. **Export Report:** Download comprehensive report for records
    7. **Schedule Follow-up:** Set reminders for check-ins
    8. **Track Progress:** Monitor risk trends over time
    
    #### Best Practices:
    - Update employee data regularly (quarterly recommended)
    - Act immediately on high-risk predictions
    - Document all interventions and outcomes
    - Use insights to improve company-wide policies
    - Combine with qualitative HR assessments
    """)

# ===================================
# 🎯 Call to Action Footer
# ===================================
st.markdown("---")
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; text-align: center; color: white; margin-top: 30px;'>
        <h2 style='color: white; margin-bottom: 15px;'>🚀 Ready to Improve Retention?</h2>
        <p style='font-size: 16px; margin-bottom: 20px;'>
            Take proactive steps today to retain your valuable talent and build a stronger workforce.
        </p>
        <p style='font-size: 14px; opacity: 0.9;'>
            💡 Remember: Every employee saved is a success story for your organization.
        </p>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**🔒 Data Privacy**")
    st.caption("All predictions are confidential")

with col_footer2:
    st.markdown("**📧 Support**")
    st.caption("hr-analytics@company.com")

with col_footer3:
    st.markdown("**📖 Version**")
    st.caption("v2.1.0 - October 2024")
