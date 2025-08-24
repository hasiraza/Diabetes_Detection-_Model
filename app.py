import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# from Dm2 import voting_clf

lr=LabelEncoder()
try:
    with open("Model/Diabetes Detection.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print("❌ Error while loading model:", e)

st.title('Diabetes Detection Model')
gender=st.selectbox("Gender",["Male","Female"])
age=st.number_input("Age")
hypertension=st.selectbox("Do you have Hypertension",["Yes","NO"])
heart_disease=st.selectbox("Do you have Heart_disease",["Yes","NO"])
smoking_history=st.selectbox("Do you smooking",["No info","Current","former","never"])
blood_glucose_level=st.number_input("Blood glucose level")
button_predict=st.button("Predict")


def prediction(gender,age, hypertension, heart_disease, smoking_history,blood_glucose_level):
    data = {
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'blood_glucose_level': [blood_glucose_level]
    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)


    # Encode the categorical columns
    categorical_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'blood_glucose_level']
    for column in categorical_columns:
        df[column] = lr.fit_transform(df[column])
    # df = ss.fit_transform(df)

    re =model.predict(df).reshape(1,-1)
    return re[0]
if button_predict:
    result= prediction(gender,age, hypertension, heart_disease, smoking_history, blood_glucose_level)
    if result == 1:
         st.write("So sad, You are serving with Daibetes")
    else:
      st.write("Congulation! You are not serving with Daibetes")