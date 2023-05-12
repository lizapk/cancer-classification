import pickle
import streamlit as st
import numpy as np

# membaca model
model = pickle.load(open('cancer.sav', 'rb'))
scaler = pickle.load(open('scaler.sav','rb'))

#judul web
st.title('Prediksi Kanker')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    radius_mean = st.number_input('input nilai Radius karakteristik visual kanker')
    texture_mean = st.number_input('input nilai Tekstur karakteristik visual kanker')
    perimeter_mean = st.number_input('input nilai Keliling karakteristik visual kanker')

with col2 :
    area_mean = st.number_input('input nilai Luas karakteristik visual kanker')
    smoothness_mean = st.number_input('input nilai Kelembutan karakteristik visual kanker')
    compactness_mean = st.number_input('input nilai Kepadatan karakteristik visual kanker')

# code untuk prediksi
prediction = ''
input_data = (radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

# membuat tombol untuk prediksi
if st.button('Test Prediksi Kanker'):
    cancer_prediction = model.predict(std_data)
    if(cancer_prediction[0] == 1):
        prediction = 'Pasien terkena Kanker'
    else:
        prediction = 'Pasien tidak terkena Kanker'
    st.success(prediction)
