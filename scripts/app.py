from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection,preprocessing,feature_selection,ensemble,linear_model,metrics,decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#dataframes to be used in the app

df=pd.read_csv('E:\Machine Learning Projects\Covid19Prediction\data\corona_tested_individuals_ver_0083.english.csv')
new_df=pd.read_csv('E:\Machine Learning Projects\Covid19Prediction\data\covid_processed.csv')

#data preparation
def predictors(cough,fever,sore_throat,shortness_of_breath,headache):
    ft_prediction=dt_clf.predict([cough,fever,sore_throat,shortness_of_breath,headache])
    if ft_prediction==0:
        pred="negative"
    else:
        pred="positive"
    return pred


features=new_df[['gender_male','gender_female','cough', 'fever', 'sore_throat', 'shortness_of_breath','head_ache','test_indication_Abroad','test_indication_Contact with confirmed','test_indication_Other']]
corona_result= new_df[['corona_result']]
X_train, X_test, y_train,y_test=train_test_split(features,corona_result,test_size=0.3)
scaler=StandardScaler()
train_features=scaler.fit_transform(X_train)
test_features=scaler.transform(X_test)
#building the model 
dt_clf=DecisionTreeClassifier()
dt_clf.fit(train_features,y_train)
train_score=dt_clf.score(train_features,y_train)
test_score=dt_clf.score(test_features,y_test)
y_predict=dt_clf.predict(test_features)

confusion=confusion_matrix(y_test,y_predict)
FN=confusion[1][0]
TN=confusion[0][0]
TP=confusion[1][1]
FP=confusion[0][1]

#title and brief description of the app's purpose
st.write(""" 
# A COVID-19 Prediction App
""")



nav=st.sidebar.radio("Navigation",["Home","Prediction"])

if nav=="Home":
    st.image("E:\Machine Learning Projects\Covid19Prediction\data\covid_image.jpg")
    st.write(""" 
        This app allows the user to input their symptoms and the app will predict whether or not the user has COVID-19.
        
        The data used in the prediction is COVID-19 data for the period March 2020 to November 2020. A sample of the data is displayed below:
        """)

    if st.checkbox("Show Data"):
        st.table(df.head())

if nav=="Prediction" : 
    st.write("""
        ### User Information 
        """)

    name=st.text_input("Enter your full names: ")
    sex=st.selectbox("Select your gender",options=["Male","Female"])
    age=st.slider("Drag the slider to input your age",0,100)
    test_indication=st.selectbox("Select the reason for the taking test",options=["Interacted with a confirmed individual","Travelled Abroad","Other reasons"])

    st.write("""
        ### Symptoms Experienced

        Check the box to select the symptoms you are experiencing
        """)

    headache=st.checkbox('Headache')
    cough=st.checkbox('Cough')
    fever=st.checkbox('Fever')
    sore_throat=st.checkbox('Sore Throat')
    shortness_of_breath=st.checkbox('Shortness_ of_breath')

    gender_male=1 if sex=='Male' else 0
    gender_female=1 if sex=='Female' else 0
    cough=1 if cough=="cough" else 0
    fever=1 if fever=='fever' else 0
    sore_throat=1 if sore_throat=='sore_throat' else 0
    shortness_of_breath=1 if shortness_of_breath=='shortness of breath' else 0
    headache=1 if headache=='headache' else 0

    test_indication_Abroad, test_indication_Contact_with_confirmed, test_indication_Other=0,0,0
    if test_indication == 'Travelled Abroad':
        test_indication_Abroad=1
    elif test_indication == 'Interacted with a confirmed individual':
        test_indication_Contact_with_confirmed=1
    else:
        test_indication_Other=1

    input_data=scaler.transform([[gender_male,gender_female,cough, fever, sore_throat, shortness_of_breath,headache,test_indication_Abroad, test_indication_Contact_with_confirmed, test_indication_Other]])
    prediction=dt_clf.predict(input_data)
    prediction_prob=dt_clf.predict_proba(input_data)

    st.write("""
    ### Do you have COVID-19??
    """)
    if st.button("The Prediction"):
        if prediction[0]==1:
            st.success("The probaility that {} has COVID is {}%" .format (name,round(prediction_prob[0][1]*100,3)))
        else:
            st.success("The probaility that {} does not have COVID is {}%" .format (name,round(prediction_prob[0][0]*100,3)))
    





    


