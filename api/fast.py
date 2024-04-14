import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from Skin_Project.ml_logic.registry import load_model
from PIL import Image
import os
from Skin_Project.ml_logic.data import get_data, resize_data, flat_images
from Skin_Project.params import *
from Skin_Project.ml_logic.registry import load_best_model
from starlette.responses import Response
import cv2
import tensorflow as tf
from pydantic import BaseModel
import pickle
import json
from typing import Annotated, Optional


class Item(BaseModel):
    sex: str
    age: int
    localization: str

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# app.state.model = load_model()

@app.get("/")
def root():
    return {
    'Test': 'This is not a test... LOL'
}

@app.post('/binary_classification')
async def custom_binary_classification(img: UploadFile=File(...)):

    contents = await img.read()
    image = np.fromstring(contents, np.uint8)
    image=cv2.imdecode(image, cv2.IMREAD_COLOR)

    local_best_model_path = f"{CHEMIN_BINARY}/best_model.h5"
    binary_model = tf.keras.models.load_model(local_best_model_path)

    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    threshold = THRESHOLD

    df_new_image =image_resized/255
    df_new_image = np.array(df_new_image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    prediction = binary_model.predict(df_new_image)

    if prediction[0][0] < float(threshold) :
        result=('It does not look dangerous')
    else:
        result=('It will be better if you go check that out')

    return Response(content=result)

@app.post('/multiclass_classification')
async def custom_multiclass_predict(img: UploadFile=File(...)):

    multi_contents = await img.read()
    multi_image = np.fromstring(multi_contents, np.uint8)
    multi_image=cv2.imdecode(multi_image, cv2.IMREAD_COLOR)

    local_best_model_path_cat = f"{CHEMIN_CAT}/best_model.h5"
    multiclass_model = tf.keras.models.load_model(local_best_model_path_cat)

    multi_image_resized = cv2.resize(multi_image, (IMAGE_SIZE, IMAGE_SIZE))

    multi_new_image = multi_image_resized / 255
    multi_new_image = np.array(multi_new_image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    multi_prediction = multiclass_model.predict(multi_new_image)
    cat_pred = np.argmax(multi_prediction[0])-1

    multiclass_dict = {4:'Melanocytic nevus: Not dangerous!', 6:'Melanoma: It will be better if you go check that out', 2:'Seborrheic keratosis: Not dangerous!', 1:'Basal cell carcinoma: It will be better if you go check that out', 0:'Actinic keratosis: It will be better if you go check that out', 5:'Vascular lesion: It will be better if you go check that out', 3:'Dermatofibrome: Not dangerous!'}
    mole_type = multiclass_dict[cat_pred]

    return Response(content=mole_type)

@app.get('/ping')
def Ping():
    return 'Pong'

@app.post('/predict_metadata')
async def custom_predict_metadata(sex: Annotated[str, Form()],
    age: Annotated[int, Form()], localization: Annotated[str, Form()],img: UploadFile = File(...),):

    #Get the image from streamlit and convert it
    contents = await img.read()
    image = np.fromstring(contents, np.uint8)
    image=cv2.imdecode(image, cv2.IMREAD_COLOR)

    local_best_model_path_cat = f"{CHEMIN_CAT}/best_model.h5"
    multiclass_model = tf.keras.models.load_model(local_best_model_path_cat)


    multi_image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    multi_new_image = multi_image_resized / 255
    multi_new_image = np.array(multi_new_image).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    cnn_prediction = multiclass_model.predict(multi_new_image)

    inputs = {
        "age": age,
        "sex": sex.lower(),
        "localization": localization.lower()
    }

    with open('/home/pavel/code/Capucine-Darteil/Skin_Images/api/preproc.pkl', 'rb') as file:
        load_preproc = pickle.load(file)

    # with open('./preproc.pkl', 'rb') as file:
    #     load_preproc = pickle.load(file)

    # Create the pandas DataFrame
    df_X = pd.DataFrame(inputs,index=[0])
    new_x = load_preproc.transform(df_X)


    meta_multiclass_model = pickle.load(open(f"{CHEMIN_META_CAT}/model_ml", 'rb'))

    y_pred_gcb = meta_multiclass_model.predict_proba(new_x)

    weight_cnn = 0.9

    y_pred = np.asarray(cnn_prediction) * weight_cnn + np.asarray(y_pred_gcb) * (1-weight_cnn)

    cat_pred = np.argmax(y_pred[0])-1

    multiclass_dict = {4:'Melanocytic nevus: Not dangerous!', 6:'Melanoma: It will be better if you go check that out', 2:'Seborrheic keratosis: Not dangerous!', 1:'Basal cell carcinoma: It will be better if you go check that out', 0:'Actinic keratosis: It will be better if you go check that out', 5:'Vascular lesion: It will be better if you go check that out', 3:'Dermatofibrome: Not dangerous!'}
    mole_type = multiclass_dict[cat_pred]

    return Response(content=mole_type)
