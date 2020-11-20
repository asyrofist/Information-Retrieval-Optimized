import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Dataset Heart")
heart_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
st.write(heart_df)

st.sidebar.header("Fitur Parameter")
genre = st.sidebar.radio("What do you choose",('extract_df','desc_df', 'feature_df', 'group_df'))
if genre == 'extract_df':
  st.subheader("Based on Deskripsi")
  list_feature = st.multiselect("Feature Select?", 
                    ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'], 
                    ['age','creatinine_phosphokinase', 
                     'ejection_fraction', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'time'])
  hasil = heart_df[list_feature]
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
  if genre_df == 'histogram':
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif genre_df == 'korelasi':
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = hasil.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)
    
    st.sidebar.subheader("Classification Parameter")
    klas_df = st.sidebar.radio("What do you choose",('dt', 'gnb', 'lr', 'rfc', 'rfr', 'svc', 'svr', 'knn'))
    list_item = st.selectbox("Feauture Select?", ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT'])
    rasio = st.sidebar.slider('What size?', 0.1, 0.6, 0.3)
    X = hasil
    y = hasil[list_item]
    st.write(X)
    st.write(y)
    
elif genre == 'desc_df':
  st.subheader("Based on Deskripsi")
  hasil = heart_df.describe()
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
  if genre_df == 'histogram':
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif genre_df == 'korelasi':
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = hasil.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)

elif genre == 'feature_df':
  st.subheader("Based on Feature")
  list_feature = st.selectbox("Berdasarkan?", ['age', 'anaemia', 'creatinine_phosphokinase', 
                                               'diabetes', 'ejection_fraction', 'high_blood_pressure', 
                                               'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
                                               'smoking', 'time', 'DEATH_EVENT'])
  list_size = st.radio("What Size?", ['max', 'min', 'average'])
  if list_size == 'max':
    hasil = heart_df[heart_df[list_feature] == heart_df[list_feature].max()] # berdasarkan maximum
    st.write(hasil)
    st.sidebar.subheader("Evaluation Parameter")
    genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
    if genre_df == 'histogram':
      st.subheader("Histogram Parameter")
      fig, ax = plt.subplots()
      ax.hist(hasil)
      st.pyplot(fig)
    elif genre_df == 'korelasi':
      st.subheader("Heatmap Correlation")
      fig, correlation = plt.subplots()
      correlation = hasil.corr()
      sns.heatmap(correlation)
      st.pyplot(fig)
  elif list_size == 'min':
    hasil = heart_df[heart_df[list_feature] == heart_df[list_feature].min()] # berdasarkan maximum
    st.write(hasil)
    st.sidebar.subheader("Evaluation Parameter")
    genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
    if genre_df == 'histogram':
      st.subheader("Histogram Parameter")
      fig, ax = plt.subplots()
      ax.hist(hasil)
      st.pyplot(fig)
    elif genre_df == 'korelasi':
      st.subheader("Heatmap Correlation")
      fig, correlation = plt.subplots()
      correlation = heart_df.corr()
      sns.heatmap(correlation)
      st.pyplot(fig)    
  elif list_size == 'average':
    hasil = heart_df[heart_df[list_feature] == heart_df[list_feature].average()] # berdasarkan maximum
    st.write(hasil)
    st.sidebar.subheader("Evaluation Parameter")
    genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
    if genre_df == 'histogram':
      st.subheader("Histogram Parameter")
      fig, ax = plt.subplots()
      ax.hist(hasil)
      st.pyplot(fig)
    elif genre_df == 'korelasi':
      st.subheader("Heatmap Correlation")
      fig, correlation = plt.subplots()
      correlation = hasil.corr()
      sns.heatmap(correlation)
      st.pyplot(fig)     
  
elif genre == 'group_df':
  st.subheader("Based on Group")
  list_feature = st.multiselect("Berdasarkan?", 
                    ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'], 
                    ['age'])
  list_select = st.selectbox("Berdasarkan?", ['age', 'anaemia', 'creatinine_phosphokinase', 
                                               'diabetes', 'ejection_fraction', 'high_blood_pressure', 
                                               'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
                                               'smoking', 'time', 'DEATH_EVENT'])
  hasil = heart_df.groupby(list_feature)[list_select].describe() # berdasarkan group
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  genre_df = st.sidebar.radio("What do you choose",('korelasi', 'histogram'))
  if genre_df == 'histogram':
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif genre_df == 'korelasi':
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = hasil.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)
