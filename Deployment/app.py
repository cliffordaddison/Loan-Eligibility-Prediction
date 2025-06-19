import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and scaler
def load_model(filename='loan_eligibility_model.pkl'):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing function
def preprocess_input(input_data):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])

    columns = [
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    categorical_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(mapping)
    
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]
    return input_data

# Streamlit App
def main():
    st.title("🏦 Loan Eligibility Predictor")
    st.write("Provide applicant information to check loan eligibility using a machine learning model.")

    # Load the model
    model_data = load_model()
    if model_data is None:
        st.stop()

    # User input
    st.subheader("Enter Applicant Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        married = st.selectbox('Married', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    with col2:        
        self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
        applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
        coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
        loan_amount = st.number_input('Loan Amount', min_value=0, value=150)
    with col3:
        loan_term = st.number_input('Loan Amount Term (in days)', min_value=0, value=360)
        credit_history = st.selectbox('Credit History', [1, 0])
        property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    if st.button('Analyze Eligibility'):
        try:
            processed_input = preprocess_input(input_data)
            scaled_input = model_data['scaler'].transform(processed_input)
            prediction = model_data['model'].predict(scaled_input)
            prediction_proba = model_data['model'].predict_proba(scaled_input)

            st.subheader("Result:")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Approval Probability", f"{prediction_proba[0][1]*100:.2f}%")
            with col2:
                st.metric("Rejection Probability", f"{prediction_proba[0][0]*100:.2f}%")

            if prediction[0] == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Not Approved")

            with st.expander("See Input Details"):
                st.write(pd.DataFrame([input_data]))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("""
This tool predicts whether a loan application is likely to be approved based on user-provided information.

**Model Used**: Logistic Regression  
**Features**: Income, Employment, Credit History, etc.

Upload by Clifford Addison
""")

# Run the app
if __name__ == '__main__':
    main()
