import os
from data import get_data, merge_dicts, resize_data, flat_images, df_final, upload_df
from Skin_Project.params import *


def load_data(CHEMIN_1,CHEMIN_2,CHEMIN_METADATA,IMAGE_SIZE):
    dict_1 = get_data(CHEMIN_1)
    dict_2 = get_data(CHEMIN_2)
    data = merge_dicts(dict_1, dict_2)
    data = resize_data(data,int(IMAGE_SIZE))
    df = flat_images(data)
    df = df_final(CHEMIN_METADATA,df)
    print(df)
    upload_df(df, IMAGE_SIZE)
    return df

if __name__ == '__main__':
    load_data(CHEMIN_1,CHEMIN_2,CHEMIN_METADATA,IMAGE_SIZE)
