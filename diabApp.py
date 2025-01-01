import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('diabetesModel.pkl')
scaler = joblib.load('scaler.pkl')

def predict_diabetes(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

def main():
    st.set_page_config(page_title='Diabetes Prediction App', layout='wide')
    st.title('Diabetes Prediction App')

    menu = ['Home', 'Diabetes Check', 'Help', 'About', 'Contact Us']
    choice = st.sidebar.selectbox('Navigation', menu)

    if choice == 'Home':
        st.write('# Welcome to the Diabetes Prediction App')
        st.write('Learn about diabetes and check your risk!')

    elif choice == 'Diabetes Check':
        st.write('# Diabetes Check')
        st.write('Fill in the details below and click Predict.')

        with st.form(key='diabetes_form'):
            st.write('Enter your details:')
            pregnancies = st.number_input('Pregnancies', min_value=0)
            glucose = st.number_input('Glucose', min_value=0)
            blood_pressure = st.number_input('Blood Pressure', min_value=0)
            skin_thickness = st.number_input('Skin Thickness', min_value=0)
            insulin = st.number_input('Insulin', min_value=0)
            bmi = st.number_input('BMI', min_value=0.0)
            diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0)
            age = st.number_input('Age', min_value=0)

            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree_function, age]])
            prediction = predict_diabetes(input_data)
            
            st.subheader('Prediction')
            result = 'Positive' if prediction == 1 else 'Negative'
            st.write(f'The prediction is: {result}')

    elif choice == 'Help':
        st.write('# Help')
        st.write("""
    Steps to use the Diabetes Prediction App:
    1. Navigate to the 'Diabetes Check' section.
    2. Enter your medical parameters.
    3. Click on 'Predict' to get your diabetes prediction and probability.
    """)
        
    elif choice == 'About':
        st.write('# About')
        st.write('This is a diabetes detection app based on')

    elif choice == 'Contact Us':
        st.write('# Contact Us')
        st.write('Contact details will be provided here.')
if __name__ == '__main__':
    main()

