# 🧠 Employee Attrition Classification

### 🎯 Objective
The goal of this project is to build a **Machine Learning Classification Model** that predicts whether an employee is likely to leave (attrition) or stay in the company.  
This project focuses on not just accuracy but also **interpretability and fairness**, providing valuable insights for HR decision-making.

---

## 📊 Dataset
**Source:** [IBM HR Analytics Employee Attrition Dataset – Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-employee-attrition-performance)

- **Rows:** 1470  
- **Columns:** 35  
- **Target Variable:** `Attrition` (Yes/No)  

This dataset contains employee demographic, compensation, and performance-related features such as:
- Age, Gender, Department, Job Role  
- Monthly Income, Work-Life Balance, Years at Company  
- Job Satisfaction, Overtime, and more.

---

## ⚙️ Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Inspected data types, distributions, and missing values.  
- Visualized attrition trends across age, department, gender, and salary levels.  
- Identified imbalanced classes (~15% attrition).  

### 2️⃣ Feature Engineering
- Encoded categorical features using **Label Encoding** and **One-Hot Encoding**.  
- Scaled numeric features using **StandardScaler**.  
- Derived meaningful new features:  
  - **Tenure Bands** (short, medium, long)  
  - **Work-Life Balance Index**  
  - **Promotion Gap**  

### 3️⃣ Model Building
Implemented and compared multiple supervised ML models:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

**Techniques used:**
- Stratified Train-Test Split  
- Class imbalance handling via **SMOTE** and **class weights**  
- Hyperparameter tuning using **GridSearchCV**

### 4️⃣ Model Evaluation
Evaluated models using:
- Confusion Matrix  
- Accuracy, Precision, Recall, F1-score  
- ROC-AUC Curve and Precision-Recall Curve  
- Business-aware cost analysis (false negatives costlier than false positives)

### 5️⃣ Explainability & Fairness
Used **SHAP** and **LIME** for model interpretability:
- Identified top influencing factors (e.g., Overtime, Job Role, Salary, Work-Life Balance)  
- Checked for fairness across gender, department, and age groups  

### 6️⃣ Deployment
The best-performing model was deployed using **Streamlit**.  
You can try the live demo below 👇  

👉 **[Employee Attrition Prediction App](https://employeeattritionclassification-gxddesvntfvnrds6hnzqef.streamlit.app/)**  

---

## 🧰 Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **XGBoost**, **LightGBM**, **CatBoost**
- **imbalanced-learn (SMOTE)**
- **SHAP**, **LIME**
- **Streamlit**

---

## 📂 Project Structure

Employee_Attrition_Classification/
│
├── data/
│ └── employee_data.csv
│
├── models/
│ └── attrition_model.pkl
│
├── Employee_Attrition_Classification.ipynb
│
└── README.md


---

## 📈 Results & Insights
- **Best Model:** Random Forest (AUC = 0.91)  
- **Key Attrition Drivers:**  
  - Overtime  
  - Job Satisfaction  
  - Monthly Income  
  - Years at Company  
  - Work-Life Balance  

### 💡 Business Recommendations
- Improve work-life balance for employees in high-risk roles.  
- Review compensation and promotion cycles.  
- Monitor long-tenured employees in sales and research departments.

---

## 👨‍💻 Author
**Uday N**  
📧 [udaynarasimhamurthy@gmail.com](mailto:udaynarasimhamurthy@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/udaynmurthy/)  
🔗 [GitHub](https://github.com/UDAYNMURTHY13)

---

## 🏁 Future Improvements
- Integrate real-time employee data via API.  
- Build HR dashboards for attrition risk monitoring.  
- Explore deep learning models for prediction.

---

⭐ *If you found this project useful, don’t forget to star the repository!* ⭐
