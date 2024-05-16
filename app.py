import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title=" Employee Attrition Prediction", layout="wide")
st.header("Attrition Prediction per Employee")

st.write("Data Source: [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)")
model = pickle.load(open('models/model_2024_05_16.sav', 'rb'))
encoders = pickle.load(open('models/encoder_2024_05_16.sav', 'rb'))
explainer = pickle.load(open('models/explainer_2024_05_16.sav', 'rb'))
col1, col2 = st.columns(2)

with col1: 
    result = st.empty()
with col2: 
    feature_importances_title = st.empty()
    feature_importances = st.empty()

active_proba, churn_proba = 0,0

result.write("<div style='padding: 10px 40px; margin-top: -20px;'>"+
                "<h5 style='text-align: center; background-color: #6750A4; padding: 10px; margin: 0 auto;'>Prediction Breakdown</h5>" +
                f"<h4 style='text-align: center;'>{active_proba * 100:.2f}%</h4>" +
                "<h6 style='text-align: center;background-color: #1D192B; padding: 10px; width:50%; margin: 0 auto;'>Will Likely Stay</h6>" + 
                f"<h4 style='text-align: center;'>{churn_proba * 100:.2f}%</h4>" +
                "<h6 style='text-align: center;background-color: #B3261E; padding: 10px; width:50%; margin: auto auto;'>Will Likely Churn</h6>" +
                "</div>", unsafe_allow_html=True)

with st.sidebar:
    generate = st.button('Generate')
    age_generation = st.selectbox(
        "Age Generation",
        ('Millennials', 'Gen X', 'Gen Z', 'Boomers II (a/k/a Generation Jones)*'))
    
    business_travel = st.selectbox(
        "Business Travel",
        ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
    
    daily_rate = st.number_input('Daily Rate', min_value=1, value="min", step=1, format='%i', key=None, label_visibility="visible")

    department = st.selectbox(
        "Department",
        ('Research & Development', 'Sales', 'Human Resources'))
    
    distance_from_home = st.selectbox(
        "Distance from Home",
        ('Short Distance', 'Medium Distance', 'Long Distance'))
    
    education = st.selectbox(
        "Education",
        ('Bachelor', 'Master', 'College', 'Below College', 'Doctor'))
    
    educationField = st.selectbox(
        "Education Field",
        ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'))
    
    environmentSatisfaction = st.selectbox(
        "Environment Satisfaction",
        ('High', 'Very High', 'Medium', 'Low'))
    
    gender = st.selectbox(
        "Gender",
        ('Male', 'Female'))
    
    job_involvement = st.selectbox(
        "Job Involvement",
        ('High', 'Medium', 'Very High', 'Low'))
    
    job_level = st.selectbox(
        "Job Involvement",
        (1,2,3,4,5))
    
    jobRole = st.selectbox(
        "Job Role",
        ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
    
    jobSatisfaction = st.selectbox(
        "Job Satisfaction",
        ('Very High', 'High', 'Low', 'Medium'))
    
    maritalStatus = st.selectbox(
        "Marital Status",
        ('Married', 'Single', 'Divorced'))

    numCompaniesWorked = st.number_input('Num Companies Worked', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")

    overTime = st.selectbox(
        "Overtime Preference",
        (0, 1))
    
    percentSalaryHike = st.number_input('Percent Salary Hike', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")

    performanceRating = st.selectbox(
        "Performance Rating",
        ('Excellent', 'Outstanding'))
    
    relationshipSatisfaction =st.selectbox(
        "Relationship Satisfaction   ",
        ('High', 'Very High', 'Medium', 'Low'))
    
    stockOptionLevel = st.selectbox(
        "Stock Option Level",
        (0, 1, 2, 3))
    
    totalWorkingYears = st.number_input('Total Working Years', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")
    
    trainingTimesLastYear = st.number_input('Training Times Last Year', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")
    
    WorkLifeBalance = st.selectbox(
        "Work Life Balance",
        ('Better', 'Good', 'Best', 'Bad'))
    
    yearsAtCompany = st.number_input('Years At Company', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")
    
    yearsInCurrentRole = st.number_input('Years In Current Role', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")
    
    yearsSinceLastPromotion = st.number_input('Years Since Last Promotion', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")
    
    yearsWithCurrManager = st.number_input('Years With Curr Manager', min_value=0, value="min", step=1, format='%i', key=None, label_visibility="visible")

if generate:
    features = ['AgeGeneration', 'BusinessTravel', 'DailyRate',
        'Department', 'DistanceFromHomeCategory',
        'Education', 'EducationField', 'EnvironmentSatisfaction',
        'Gender', 'JobInvolvement',
        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
            'Overtime_Preference', 'PercentSalaryHike',
        'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager']
    target = 'Attrition'

    values_to_predict = [age_generation, business_travel, daily_rate, department, distance_from_home, education, educationField, 
                            environmentSatisfaction, gender, job_involvement, job_level, jobRole, jobSatisfaction, maritalStatus,
                            numCompaniesWorked, overTime, percentSalaryHike, performanceRating, relationshipSatisfaction, 
                            stockOptionLevel, totalWorkingYears, trainingTimesLastYear, WorkLifeBalance, yearsAtCompany,
                            yearsInCurrentRole, yearsSinceLastPromotion, yearsWithCurrManager]

    df_to_predict = pd.DataFrame(values_to_predict, index=features).T

    for col in encoders.keys():
        if (col != 'Attrition') and (col != 'OverTime'):
            df_to_predict[col] = encoders[col].transform(df_to_predict[col])

    pred_proba = model.predict_proba(df_to_predict)
    active_proba = float(pred_proba[0][0])
    churn_proba = float(pred_proba[0][1])

    shap_values = explainer.shap_values(df_to_predict)

    shap_values_abs = np.abs(shap_values).mean(axis=0)
    df_feature_importance = pd.DataFrame(shap_values_abs, index=features, columns=['feature_importance'])
    df_feature_importance.sort_values(by='feature_importance', ascending=False, inplace=True)

    feature_importances_str = "<div><ol>"
    for index in features:
        feature_importances_str += f"<li>{index}: {df_feature_importance.loc[index].values[0]*100:.2f}%\n</li>"

    result.write("<div style='padding: 10px 40px; margin-top: -20px;'>"+
                "<h5 style='text-align: center; background-color: #6750A4; padding: 10px; margin: 0 auto;'>Prediction Breakdown</h5>" +
                f"<h4 style='text-align: center;'>{active_proba * 100:.2f}%</h4>" +
                "<h6 style='text-align: center;background-color: #1D192B; padding: 10px; width:50%; margin: 0 auto;'>Will Likely Stay</h6>" + 
                f"<h4 style='text-align: center;'>{churn_proba * 100:.2f}%</h4>" +
                "<h6 style='text-align: center;background-color: #B3261E; padding: 10px; width:50%; margin: auto auto;'>Will Likely Churn</h6>" +
                "</div>", unsafe_allow_html=True)

    feature_importances_title.title("Factors for Attrition")
    feature_importances.write(feature_importances_str + "</ol></div>")