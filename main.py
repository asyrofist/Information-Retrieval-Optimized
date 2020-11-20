import sreamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.head("Dataset Heart")
heart_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
st.write(heart_df)

desc          = st.sidbear.checkbox("berdasarkan deskripsi")
df_feature    = st.sidebar.checbox("berdasarkan fitur")
group_feature = st.sidebar.checkbox("berdasarkan group")
if desc:
  hasil = heart_df.describe()
  st.write(hasil)
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    hasil.hist()
  elif korelasi:
    correlation = heart_df.corr()
    sns.heatmap(corelation)

    
elif df_feature:
  list_feature = st.sidebar.selectbox("Berdasarkan?", ['max', 'min', 'average'])
  hasil = heart_df[heart_df.anaemia == heart_df.anaemia.list_feature()] # berdasarkan maximum
  st.write(hasil)
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    hasil.hist()
  elif korelasi:
    correlation = heart_df.corr()
    sns.heatmap(corelation)
    
elif group_feature:
  list_feature = st.sidebar.selectbox("Berdasarkan?", ['age	anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
  data_feature = st.sidebar.selectbox("Berdasarkan?", ['age	anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
  hasil = heart_df.groupby(data_feature).list_feature.describe() # berdasarkan group
  st.write(hasil)
  histogram = st.sidebar.checkbox("histogram")
  korelasi = st.sidebar.checkbox("korelasi")
  if histogram:
    hasil.hist()
  elif korelasi:
    correlation = heart_df.corr()
    sns.heatmap(corelation)
