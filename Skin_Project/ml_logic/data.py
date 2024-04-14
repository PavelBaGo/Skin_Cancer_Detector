import pandas as pd
import numpy as np
from skimage import io
from pathlib import Path
import cv2
import os
from Skin_Project.params import *

# Exemple chemin  : '/home/auguste/code/Capucine-Darteil/Skin_Images/raw_data/HAM10000_images_part_1'

def get_data(chemin):

    image_dir_path = '.'
    paths1 = [path.parts[-1:] for path in
         Path(chemin).rglob('*.jpg')]

    data = {}
    for i in range(len(paths1)):
        data[paths1[i][0][:-4]] = io.imread(f'{chemin}/{paths1[i][0]}')
    return data

# Concatener les dictionnaires d'images. Un dictionnaire par dossier avec des images.
def merge_dicts(dict_1, dict_2):
    dict_1.update(dict_2)
    return dict_1

# Modifier la taille des images
def resize_data(data,IMAGE_SIZE):
    resized_data = {}
    for key, image in data.items():
        resized_image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        resized_data[key] = resized_image

    return resized_data

# Reshape data et transformer en DataFrame:
def flat_images(data):
    flat_data = {}
    for key, image in data.items():
        flat_image = image.flatten()
        flat_data[key] = flat_image

    df_images = pd.DataFrame(flat_data).transpose()
    return df_images

def df_final(CHEMIN_METADATA, df_images):
    metadf = pd.read_csv(CHEMIN_METADATA)
    metadf = metadf.set_index('image_id')
    df = pd.concat([metadf,df_images],axis=1)
    df = df.reset_index(drop=False)
    return df


def upload_df(df, IMAGE_SIZE):
    df.to_csv(f'df_complet_{IMAGE_SIZE}x{IMAGE_SIZE}.csv', index=False)
    pass
