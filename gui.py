import streamlit as st
import joblib

model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('E-commerce Customer Yearly Expenditure Prediction')

avg_session_length = st.number_input('Average Session Length (in minutes)', min_value=0.0, max_value=60.0, value=33.0, step=0.1)
time_on_app = st.number_input('Time on App (in minutes)', min_value=0.0, max_value=60.0, value=12.0, step=0.1)
time_on_website = st.number_input('Time on Website (in minutes)', min_value=0.0, max_value=60.0, value=37.0, step=0.1)
length_of_membership = st.number_input('Length of Membership (in years)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)

if st.button('Predict Yearly Expenditure'):
    user_input_scaled = scaler.transform([[avg_session_length, time_on_app, time_on_website, length_of_membership]])    
    prediction = model.predict(user_input_scaled)
    if(prediction[0]<0):
        st.success(f'The predicted yearly expenditure is: ${0.00}')
    else:
        st.success(f'The predicted yearly expenditure is: ${prediction[0]:.2f}')

