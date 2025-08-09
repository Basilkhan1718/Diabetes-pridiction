import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")

st.markdown("This app predicts diabetes using a trained Random Forest model.")

# Input fields for the 8 features
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Tolerance Test Result", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 10, 100, 33)

# Prediction
if st.button("Predict"):
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, pedigree, age]])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
         st.error(f"⚠️ The person is likely **diabetic** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The person is likely **not diabetic** (Probability: {probability:.2f})")
