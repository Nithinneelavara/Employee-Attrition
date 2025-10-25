import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("🚀 Employee Attrition Prediction App")
st.write("Upload a CSV file for batch predictions or enter details for a single employee.")

# Load model and scaler once
try:
    loaded_model = joblib.load("models/rf_attrition_model.pkl")
    loaded_scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check the path.")

# Columns in the order used during training
trained_columns = [
    "Age","BusinessTravel","DailyRate","Department","DistanceFromHome",
    "Education","EducationField","EnvironmentSatisfaction","Gender","HourlyRate",
    "JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus",
    "MonthlyIncome","MonthlyRate","NumCompaniesWorked","OverTime",
    "PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
    "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion",
    "YearsWithCurrManager","WorkLifeBalance"
]

# ------------------ Batch Prediction ------------------
uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

if uploaded_file is not None and 'loaded_model' in locals() and 'loaded_scaler' in locals():
    batch_df = pd.read_csv(uploaded_file)
    batch_df.columns = batch_df.columns.str.strip()
    
    # Ensure correct order
    batch_df = batch_df[trained_columns]

    # Convert to NumPy array to avoid feature name mismatch
    batch_array = batch_df.to_numpy()
    batch_scaled = loaded_scaler.transform(batch_array)

    # Predict
    batch_predictions = loaded_model.predict(batch_scaled)
    batch_probs = loaded_model.predict_proba(batch_scaled)

    # Attach results
    batch_df['Prediction'] = ['Leave' if p==1 else 'Stay' for p in batch_predictions]
    batch_df['Prob_Stay'] = batch_probs[:,0]
    batch_df['Prob_Leave'] = batch_probs[:,1]

    st.subheader("📊 Batch Predictions")
    st.dataframe(batch_df)

    # Individual bar charts for each employee with unique keys
    st.subheader("📊 Batch Prediction Charts")
    for i, row in batch_df.iterrows():
        st.markdown(f"**Employee {i+1} Prediction: {row['Prediction']}**")
        fig = go.Figure(data=[go.Bar(
            x=['Stay', 'Leave'],
            y=[row['Prob_Stay'], row['Prob_Leave']],
            text=[f"{row['Prob_Stay']*100:.1f}%", f"{row['Prob_Leave']*100:.1f}%"],
            textposition='auto',
            marker_color=['green','red']
        )])
        fig.update_layout(yaxis=dict(range=[0,1]), height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"batch_chart_{i}")

    # Download batch predictions
    st.download_button(
        "Download Batch Predictions CSV",
        batch_df.to_csv(index=False),
        file_name="batch_predictions.csv",
        mime="text/csv"
    )

# ------------------ Single Employee Prediction ------------------
elif 'loaded_model' in locals() and 'loaded_scaler' in locals():
    with st.form(key="employee_form"):
        st.subheader("🧑 Personal & Work Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=60, value=30)
            daily_rate = st.number_input("Daily Rate", value=1000)
            distance_from_home = st.number_input("Distance From Home", value=5)
            hourly_rate = st.number_input("Hourly Rate", value=50)
            monthly_income = st.number_input("Monthly Income", value=5000)
        with col2:
            monthly_rate = st.number_input("Monthly Rate", value=15000)
            num_companies_worked = st.number_input("Num Companies Worked", value=2)
            percent_salary_hike = st.number_input("Percent Salary Hike", value=12)
            stock_option_level = st.number_input("Stock Option Level", value=1)
            total_working_years = st.number_input("Total Working Years", value=6)
        with col3:
            training_times_last_year = st.number_input("Training Times Last Year", value=2)
            years_at_company = st.number_input("Years at Company", value=4)
            years_in_current_role = st.number_input("Years in Current Role", value=2)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", value=1)
            years_with_curr_manager = st.number_input("Years With Current Manager", value=2)

        st.subheader("🏢 Job & Organizational Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", [0, 1])
            overtime = st.selectbox("OverTime", [0, 1])
            job_level = st.number_input("Job Level", value=1)
            job_role = st.number_input("Job Role (encoded)", value=1)
            marital_status = st.selectbox("Marital Status", [1, 2, 3])
        with col2:
            business_travel = st.number_input("Business Travel (encoded)", value=1)
            department = st.number_input("Department (encoded)", value=1)
            education = st.number_input("Education", value=3)
            education_field = st.number_input("Education Field (encoded)", value=1)
            environment_satisfaction = st.number_input("Environment Satisfaction", value=3)
        with col3:
            job_involvement = st.number_input("Job Involvement", value=3)
            job_satisfaction = st.number_input("Job Satisfaction", value=3)
            performance_rating = st.number_input("Performance Rating", value=3)
            relationship_satisfaction = st.number_input("Relationship Satisfaction", value=3)
            work_life_balance = st.number_input("Work Life Balance", value=3)

        submit_button = st.form_submit_button(label="Predict Attrition")

    if submit_button:
        new_employee = [[
            age, business_travel, daily_rate, department, distance_from_home,
            education, education_field, environment_satisfaction, gender, hourly_rate,
            job_involvement, job_level, job_role, job_satisfaction, marital_status,
            monthly_income, monthly_rate, num_companies_worked, overtime,
            percent_salary_hike, performance_rating, relationship_satisfaction,
            stock_option_level, total_working_years, training_times_last_year,
            years_at_company, years_in_current_role, years_since_last_promotion,
            years_with_curr_manager, work_life_balance
        ]]

        new_employee_scaled = loaded_scaler.transform(np.array(new_employee))
        prediction = loaded_model.predict(new_employee_scaled)[0]
        prob = loaded_model.predict_proba(new_employee_scaled)[0]

        st.subheader("📊 Prediction Result (Single Employee)")
        st.markdown(f"""
        <div style="font-size:28px; font-weight:bold;">
        Prediction: {'Leave' if prediction==1 else 'Stay'}
        </div>
        <div style="font-size:20px; margin-top:10px;">
        Probability Stay: {prob[0]:.2f} <br>
        Probability Leave: {prob[1]:.2f}
        </div>
        """, unsafe_allow_html=True)

        # Unique key for single employee chart
        fig = go.Figure(data=[go.Bar(
            x=['Stay', 'Leave'],
            y=[prob[0], prob[1]],
            text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
            textposition='auto',
            marker_color=['green', 'red']
        )])
        fig.update_layout(title_text='Attrition Probability', yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True, key="single_employee_chart")
