import streamlit as st
import requests
import os
from PIL import Image
from Skin_Project.params import *
import numpy as np
from io import StringIO
import json

def set_background_color(color):
    hex_color = f"#{color}"
    custom_css = f"""
        <style>
            .stApp {{
                background-color: {hex_color};
            }}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Set background color
set_background_color("eae8fa")

st.markdown("<h1 style='text-align: center; color: #A0325B;'>Mole type detector</h1>", unsafe_allow_html=True)

st.markdown("""<style>.subheader {font-size:20px !important;}</style>""", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color: #5763CF;"class="subheader", >⚠️ Disclaimer : this tool is not intended to replace the expertise of a doctor ⚠️</p>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        background-color: skyblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""<style>.subheader_2 {font-size:15px !important;}</style>""", unsafe_allow_html=True)
st.markdown('<p style="class="subheader_2", >Please upload your mole photo</p>', unsafe_allow_html=True)

img_file_buffer = st.file_uploader('Upload an image',type=["png", "jpg", "jpeg"])
if img_file_buffer is not None :
    image = Image.open(img_file_buffer)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(image, caption="Here's the image you uploaded ☝️", use_column_width=True)


if img_file_buffer is not None:
    if st.button('Is it dangerous?'):
        with st.spinner("Wait for it..."):
                img_bytes = img_file_buffer.getvalue()
                res = requests.post(API_URL + "/binary_classification", files={'img': img_bytes})

                if res.status_code == 200:
            ### Display the image returned by the API
                    final = res.content
                    st.markdown(final.decode('utf-8'))
                else:
                    st.markdown("**Oops**, something went wrong 😓 Please try again.")
                    print(res.status_code, res.content)

    if st.button('What kind of mole type it looks like?'):
        with st.spinner("Wait for it..."):
                    img_bytes = img_file_buffer.getvalue()
                    res = requests.post(API_URL + "/multiclass_classification", files={'img': img_bytes})

                    if res.status_code == 200:
                ### Display the image returned by the API
                        final = res.content
                        st.markdown(final.decode('utf-8'))
                        # st.markdown(final)
                    else:
                        st.markdown("**Oops**, something went wrong 😓 Please try again.")
                        print(res.status_code, res.content)

sex = st.selectbox('What is your sex:',('Female', 'Male'))
age = st.number_input('What is your age? (Please put an age between 1 and 99 years):', key=int,min_value=1, max_value=199)
localization = st.selectbox('Where is the mole located on your body?:', ('Lower extremity', 'Trunk', 'Upper extremity', 'Scalp', 'Abdomen', 'Ear', 'Back', 'Face','Chest','Foot','Neck','Scalp','Hand','Genital','Acral'))

inputs = {"sex": sex, "age": age, "localization": localization}

if st.button("Analyze Mole"):
    if img_file_buffer is not None:
        img_bytes = img_file_buffer.getvalue()

        # Send image and data to API
        files = {'image': img_file_buffer}
        response = requests.post(API_URL + "/predict_metadata", data=inputs, files={'img': img_bytes})
        st.markdown(response.content.decode('utf-8'))
