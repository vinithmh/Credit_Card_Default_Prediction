import streamlit as st
import pickle
import pandas as pd

# Load your saved LightGBM model
loaded_model = pickle.load(open('final_LGBM_model.sav', 'rb'))

# Define your prediction function
def predict_credit_approval(credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status):
    input_data = preprocess_input(credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status)
    prediction_result = loaded_model.predict(input_data)
    return "Approved" if prediction_result == 1 else "Not Approved"

def preprocess_input(credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status):
    input_data = pd.DataFrame({
        'LIMIT_BAL': [credit_limit],
        'SEX': [gender],
        'AGE': [age],
        'EDUCATION_graduate school': [1 if education == 'Graduate School' else 0],
        'EDUCATION_high school': [1 if education == 'High School' else 0],
        'EDUCATION_university': [1 if education == 'University' else 0],
        'MARRIAGE_married': [1 if marital_status == 'Married' else 0],
        'MARRIAGE_single': [1 if marital_status == 'Single' else 0],
        'BILL_AMT_SEPT': [bill_amounts['BILL_AMT_SEPT']],
        'BILL_AMT_AUG': [bill_amounts['BILL_AMT_AUG']],
        'BILL_AMT_JUL': [bill_amounts['BILL_AMT_JUL']],
        'BILL_AMT_JUN': [bill_amounts['BILL_AMT_JUN']],
        'BILL_AMT_MAY': [bill_amounts['BILL_AMT_MAY']],
        'BILL_AMT_APR': [bill_amounts['BILL_AMT_APR']],
        'PAY_AMT_SEPT': [payment_amounts['PAY_AMT_SEPT']],
        'PAY_AMT_AUG': [payment_amounts['PAY_AMT_AUG']],
        'PAY_AMT_JUL': [payment_amounts['PAY_AMT_JUL']],
        'PAY_AMT_JUN': [payment_amounts['PAY_AMT_JUN']],
        'PAY_AMT_MAY': [payment_amounts['PAY_AMT_MAY']],
        'PAY_AMT_APR': [payment_amounts['PAY_AMT_APR']],
        'PAY_SEPT_-1': [1 if payment_status['PAY_SEPT'] == "-1" else 0],
        'PAY_SEPT_0': [1 if payment_status['PAY_SEPT'] == "0" else 0],
        'PAY_SEPT_1': [1 if payment_status['PAY_SEPT'] == "1" else 0],
        'PAY_SEPT_2': [1 if payment_status['PAY_SEPT'] == "2" else 0],
        'PAY_SEPT_3': [1 if payment_status['PAY_SEPT'] == "3" else 0],
        'PAY_SEPT_4': [1 if payment_status['PAY_SEPT'] == "4" else 0],
        'PAY_SEPT_5': [1 if payment_status['PAY_SEPT'] == "5" else 0],
        'PAY_SEPT_6': [1 if payment_status['PAY_SEPT'] == "6" else 0],
        'PAY_SEPT_7': [1 if payment_status['PAY_SEPT'] == "7" else 0],
        'PAY_SEPT_8': [1 if payment_status['PAY_SEPT'] == "8" else 0],
        'PAY_AUG_-1': [1 if payment_status['PAY_AUG'] == "-1" else 0],
        'PAY_AUG_0': [1 if payment_status['PAY_AUG'] == "0" else 0],
        'PAY_AUG_1': [1 if payment_status['PAY_AUG'] == "1" else 0],
        'PAY_AUG_2': [1 if payment_status['PAY_AUG'] == "2" else 0],
        'PAY_AUG_3': [1 if payment_status['PAY_AUG'] == "3" else 0],
        'PAY_AUG_4': [1 if payment_status['PAY_AUG'] == "4" else 0],
        'PAY_AUG_5': [1 if payment_status['PAY_AUG'] == "5" else 0],
        'PAY_AUG_6': [1 if payment_status['PAY_AUG'] == "6" else 0],
        'PAY_AUG_7': [1 if payment_status['PAY_AUG'] == "7" else 0],
        'PAY_AUG_8': [1 if payment_status['PAY_AUG'] == "8" else 0],
        'PAY_JUL_-1': [1 if payment_status['PAY_JUL'] == "-1" else 0],
        'PAY_JUL_0': [1 if payment_status['PAY_JUL'] == "0" else 0],
        'PAY_JUL_1': [1 if payment_status['PAY_JUL'] == "1" else 0],
        'PAY_JUL_2': [1 if payment_status['PAY_JUL'] == "2" else 0],
        'PAY_JUL_3': [1 if payment_status['PAY_JUL'] == "3" else 0],
        'PAY_JUL_4': [1 if payment_status['PAY_JUL'] == "4" else 0],
        'PAY_JUL_5': [1 if payment_status['PAY_JUL'] == "5" else 0],
        'PAY_JUL_6': [1 if payment_status['PAY_JUL'] == "6" else 0],
        'PAY_JUL_7': [1 if payment_status['PAY_JUL'] == "7" else 0],
        'PAY_JUL_8': [1 if payment_status['PAY_JUL'] == "8" else 0],
        'PAY_JUN_-1': [1 if payment_status['PAY_JUN'] == "-1" else 0],
        'PAY_JUN_0': [1 if payment_status['PAY_JUN'] == "0" else 0],
        'PAY_JUN_1': [1 if payment_status['PAY_JUN'] == "1" else 0],
        'PAY_JUN_2': [1 if payment_status['PAY_JUN'] == "2" else 0],
        'PAY_JUN_3': [1 if payment_status['PAY_JUN'] == "3" else 0],
        'PAY_JUN_4': [1 if payment_status['PAY_JUN'] == "4" else 0],
        'PAY_JUN_5': [1 if payment_status['PAY_JUN'] == "5" else 0],
        'PAY_JUN_6': [1 if payment_status['PAY_JUN'] == "6" else 0],
        'PAY_JUN_7': [1 if payment_status['PAY_JUN'] == "7" else 0],
        'PAY_JUN_8': [1 if payment_status['PAY_JUN'] == "8" else 0],
        'PAY_MAY_-1': [1 if payment_status['PAY_MAY'] == "-1" else 0],
        'PAY_MAY_0': [1 if payment_status['PAY_MAY'] == "0" else 0],
        'PAY_MAY_2': [1 if payment_status['PAY_MAY'] == "2" else 0],
        'PAY_MAY_3': [1 if payment_status['PAY_MAY'] == "3" else 0],
        'PAY_MAY_4': [1 if payment_status['PAY_MAY'] == "4" else 0],
        'PAY_MAY_5': [1 if payment_status['PAY_MAY'] == "5" else 0],
        'PAY_MAY_6': [1 if payment_status['PAY_MAY'] == "6" else 0],
        'PAY_MAY_7': [1 if payment_status['PAY_MAY'] == "7" else 0],
        'PAY_MAY_8': [1 if payment_status['PAY_MAY'] == "8" else 0],
        'PAY_APR_-1': [1 if payment_status['PAY_APR'] == "-1" else 0],
        'PAY_APR_0': [1 if payment_status['PAY_APR'] == "0" else 0],
        'PAY_APR_2': [1 if payment_status['PAY_APR'] == "2" else 0],
        'PAY_APR_3': [1 if payment_status['PAY_APR'] == "3" else 0],
        'PAY_APR_4': [1 if payment_status['PAY_APR'] == "4" else 0],
        'PAY_APR_5': [1 if payment_status['PAY_APR'] == "5" else 0],
        'PAY_APR_6': [1 if payment_status['PAY_APR'] == "6" else 0],
        'PAY_APR_7': [1 if payment_status['PAY_APR'] == "7" else 0],
        'PAY_APR_8': [1 if payment_status['PAY_APR'] == "8" else 0],
    })

    input_data = pd.get_dummies(input_data, columns=['SEX'])
    return input_data

def create_input_variables():
    st.sidebar.header("User Information")
    credit_limit = st.sidebar.number_input("Credit Limit", min_value=0)
    gender = st.sidebar.radio("Gender", ['Male', 'Female'])
    age = st.sidebar.number_input("Age", min_value=0, max_value=120)

    st.sidebar.header("Education and Marital Status")
    education = st.sidebar.selectbox("Education", ['Graduate School', 'High School', 'University'])
    marital_status = st.sidebar.selectbox("Marital Status", ['Married', 'Single'])

    # Create columns for Bill Amounts, Payment Amounts, and Payment Status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Bill Amounts")
        bill_amount_features = ['BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY', 'BILL_AMT_APR']
        bill_amounts = {}
        for feature in bill_amount_features:
            bill_amounts[feature] = st.number_input(f"Enter {feature}:", min_value=0)

    with col2:
        st.markdown("#### Payment Amounts")
        payment_amount_features = ['PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR']
        payment_amounts = {}
        for feature in payment_amount_features:
            payment_amounts[feature] = st.number_input(f"Enter {feature}:", min_value=0)

    with col3:
        st.markdown("#### Payment Status")
        payment_status_features = ['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']
        payment_status = {}
        for feature in payment_status_features:
            payment_status[feature] = st.selectbox(f"Select {feature}:", ["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"])

    return credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status


def main():
    st.title("Credit Approval Prediction")

    credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status = create_input_variables()

    if st.button("Predict Credit Approval"):
        prediction_result = predict_credit_approval(credit_limit, gender, age, education, marital_status, bill_amounts, payment_amounts, payment_status)
        st.write(f"Credit Approval Prediction: {prediction_result}")

if __name__ == "__main__":
    main()
