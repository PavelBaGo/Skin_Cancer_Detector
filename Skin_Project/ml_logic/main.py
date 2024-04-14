from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from Skin_Project.ml_logic.model import initialize_dumb_model, compile_model, train_model, evaluate_model, initialize_model
from Skin_Project.ml_logic.preprocess import labelize, sampler, drop_columns, categorize
from Skin_Project.params import *

def preproc(df_sample, dx):
    preproc = make_column_transformer(
        (FunctionTransformer(lambda x: x/255., feature_names_out='one-to-one'), list(df_sample.drop(columns=dx).columns.values)),
        remainder ='passthrough')
    return preproc


def train():
    df = pd.read_csv(CHEMIN_3, index_col=0)
    #df = categorize(df)
    #print('df categorized')
    #df = drop_columns(df)
    #print('columns dropped')
    df_labelized= labelize(df)
    print('labelized ok')

    df_sample = sampler(df_labelized)
    print (f'df sampled with a ratio of {df_sample.dx.value_counts()[0]/df_sample.shape[0]}')

    preprocess = preproc(df_sample, 'dx')
    df_processed = pd.DataFrame(preprocess.fit_transform(df_sample), columns = preprocess.get_feature_names_out())

    X_processed = df_processed.drop(columns='remainder__dx')
    y_processed=df_processed['remainder__dx']
    print('data in processing...')

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.33, random_state=42)
    print('data split')

    X_train = np.array(X_train).reshape(len(X_train), IMAGE_SIZE, IMAGE_SIZE, 3)
    X_test = np.array(X_test).reshape(len(X_test), IMAGE_SIZE, IMAGE_SIZE, 3)
    print('data reshaped :)')

    model = initialize_model()
    print('model initialized...')

    model = compile_model(model)
    print('model compiled')

    model, history = train_model(model, X_train,y_train)
    print('model trained!')

    metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)

    #print(f'loss is {metrics["loss"]}')
    #print(f'accuracy is {metrics["accuracy"]}')
    #print(f'recall is {metrics["recall"]}')
    #print(f'precision is {metrics["precision"]}')
    print(metrics)
    return metrics


if __name__ == '__main__':
    train()
