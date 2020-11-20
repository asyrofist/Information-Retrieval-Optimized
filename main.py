import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Dataset Heart")
heart_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
st.write(heart_df)

st.sidebar.header("Fitur Parameter")
desc          = st.sidebar.checkbox("berdasarkan deskripsi")
df_feature    = st.sidebar.checkbox("berdasarkan fitur")
group_feature = st.sidebar.checkbox("berdasarkan group")
if desc:
  hasil = heart_df.describe()
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif korelasi:
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = heart_df.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)
    

elif df_feature:
  hasil = heart_df[heart_df.age == heart_df.age.max()] # berdasarkan maximum
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif korelasi:
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = heart_df.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)
    
elif group_feature:
  list_feature = st.selectbox("Berdasarkan?", ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
  hasil = heart_df.groupby(list_feauture).age.describe() # berdasarkan group
  st.write(hasil)
  st.sidebar.subheader("Evaluation Parameter")
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    st.subheader("Histogram Parameter")
    fig, ax = plt.subplots()
    ax.hist(hasil)
    st.pyplot(fig)
  elif korelasi:
    st.subheader("Heatmap Correlation")
    fig, correlation = plt.subplots()
    correlation = heart_df.corr()
    sns.heatmap(correlation)
    st.pyplot(fig)
