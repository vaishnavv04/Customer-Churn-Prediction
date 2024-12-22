import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pickle

model = load_model('model.h5')

with open('onehot_geo.pkl','rb') as file :
    onehot_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file :
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file :
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',onehot_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)

credit_score = st.number_input('Credit Score')
age = st.slider('Age', min_value=18, max_value=100)
tenure = st.number_input('Tenure', min_value=0, max_value=10)
balance = st.number_input('Balance')
num_products = st.number_input('Number of Products', min_value=1, max_value=4)
estimated_salary = st.number_input('Estimated Salary')

has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))

# Create dictionary with all inputs
user_input = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Create DataFrame with user input
user_input_df = pd.DataFrame(user_input)
user_input_df = pd.concat([user_input_df, geo_df], axis=1)

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Predict churn
prediction = model.predict(user_input_scaled)
prob = prediction[0][0]

#print probability
st.write('Probability of Churn:', prob)

# Display prediction
if prob > 0.5:
    st.write('Prediction: Customer will churn')
else:
    st.write('Prediction: Customer will not churn')
