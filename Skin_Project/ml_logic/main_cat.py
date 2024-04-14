from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from Skin_Project.ml_logic.model_cat import compile_model, train_model, evaluate_model, initialize_model, initialize_fit_model_ml
from Skin_Project.ml_logic.preprocess import labelize, sampler, drop_columns, categorize
from keras.utils import to_categorical
from Skin_Project.params import *
from Skin_Project.ml_logic.registry import save_model, load_model, load_best_model, mlflow_transition_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preproc(df_sample, dx):
    preproc = make_column_transformer(
        (FunctionTransformer(lambda x: x/255., feature_names_out='one-to-one'), list(df_sample.drop(columns=dx).columns.values)),
        remainder ='passthrough')
    return preproc

#split dataset to send to different preprocs
def dataset_clean_split(df):
    df128_metadata_dropped = df.drop(columns =['lesion_id', 'dx_type'], axis=1)
    df128_cat = df128_metadata_dropped[['age','sex','localization','dx']]
    df128_pixels = df128_metadata_dropped.drop(columns = ['age','sex', 'localization'])
    return df128_cat, df128_pixels

#minmax scale age and ohe sex and localization
def preproc_metadata():
    preproc_metadata = make_column_transformer(
        (MinMaxScaler(), ['age']),
        (OneHotEncoder(sparse_output=False), ['sex', 'localization']),
        remainder='passthrough')
    return preproc_metadata


def preprocess():
    print('coucou')

    df = pd.read_csv(CHEMIN_3, index_col=0)


    #df = categorize(df)
    #print('df categorized')
    #df = drop_columns(df)
    #print('columns dropped')


    if SAMPLE_SIZE != 1.0 :
        df = sampler(df)
        print (f'df sampled with a ratio of {df.dx.value_counts()[0]/df.shape[0]}')


    if METADATA == 'yes':

        #categorize
        df = categorize(df)
        print('dataframe categorized...')

        df.reset_index(inplace=True)
        print('index reset ...')

        df = df.drop(columns='index')
        print('index dropped')

        if CLASSIFICATION == 'binary':
            df= labelize(df)
            print('labelized ok')

        #drop nan values
        df = df.dropna()
        print('NaN removed')

        #split dataset into cat and pixel datasets
        df128_cat, df128_pixels = dataset_clean_split(df)
        print('dataframe split...')

        # Definition of X and y
        X_cat = df128_cat.drop(columns='dx')
        y_cat = df128_cat.dx
        X_pixel = df128_pixels.drop(columns='dx')
        y_pixel = df128_pixels.dx
        print('dataframe DEFINETELY split...')

        #ohe and minmax features on cat dataset
        X_cat_processed = pd.DataFrame(preproc_metadata().fit_transform(X_cat))
        print('data cat in processing...')

        #standardize pixels on pixel dataset
        X_pixel_processed = pd.DataFrame(preproc(df128_pixels,'dx').fit_transform(X_pixel))
        print('data pixels in processing...')

        #Train test split on cat dataset
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat_processed, y_cat, test_size=0.33, random_state=42)
        print('cat data split')

        #Train test split on pixel dataset
        X_train_pixel, X_test_pixel, y_train_pixel, y_test_pixel = train_test_split(X_pixel_processed, y_pixel, test_size=0.33, random_state=42)
        print('pixel data split')

        #reshape pixels for CNN
        X_train_pixel = np.array(X_train_pixel).reshape(len(X_train_pixel), IMAGE_SIZE, IMAGE_SIZE, 3)
        X_test_pixel = np.array(X_test_pixel).reshape(len(X_test_pixel), IMAGE_SIZE, IMAGE_SIZE, 3)
        print('data reshaped :)')

        if CLASSIFICATION=='cat':
        ### Encoding the labels
            y_train_pixel = to_categorical(y_train_pixel, 7)
            y_test_pixel = to_categorical(y_test_pixel, 7)
            print('target pixel categorized')

        return X_train_pixel, X_test_pixel, X_train_cat, X_test_cat, y_train_pixel, y_test_pixel, y_train_cat, y_test_cat

    if METADATA == 'no':
        if CLASSIFICATION == 'binary':
            df= labelize(df)
            print('labelized ok')

        preprocess = preproc(df, 'dx')
        df_processed = pd.DataFrame(preprocess.fit_transform(df), columns = preprocess.get_feature_names_out())

        X_processed = df_processed.drop(columns='remainder__dx')
        y_processed=df_processed['remainder__dx']
        print('data in processing...')

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.33, random_state=42)
        print('data split')

        X_train = np.array(X_train).reshape(len(X_train), IMAGE_SIZE, IMAGE_SIZE, 3)
        X_test = np.array(X_test).reshape(len(X_test), IMAGE_SIZE, IMAGE_SIZE, 3)
        print('data reshaped :)')

        if CLASSIFICATION=='cat':
            ### Encoding the labels
            y_train = to_categorical(y_train, 7)
            y_test = to_categorical(y_test, 7)
            print('')

    return X_train, X_test, y_train, y_test


def train():
    if METADATA == 'no':
        X_train, X_test, y_train, y_test = preprocess()

        model = initialize_model()
        print('model initialized...')

        model = compile_model(model)
        print('model compiled')

        model, history = train_model(model, X_train,y_train)
        print('model trained!')


        if MODEL_TARGET == 'local':
            #Load the best model
            best_model = load_best_model()

            if best_model == None:
                if CLASSIFICATION == 'binary':
                    best_model_path = f"{CHEMIN_BINARY}/best_model.h5"
                    model.save(best_model_path)
                    print("First model is saved as best model !")
                    pass
                if CLASSIFICATION == 'cat':
                    best_model_path = f"{CHEMIN_CAT}/best_model.h5"
                    model.save(best_model_path)
                    print("First model is saved as best model !")
                    pass

            #best_metrics = evaluate_model(best_model, X_test, y_test,threshold=THRESHOLD)
            #print(f'ancient metrics are : {best_metrics}')

            metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)
            print(f'new metrics are : {metrics}')

            keys_ =list(metrics.keys())

            #if metrics[keys_[2]]>best_metrics['Recall'] and metrics[keys_[1]]>0.5:
            if CLASSIFICATION == 'binary':
                best_model_path = f"{CHEMIN_BINARY}/best_model.h5"
                model.save(best_model_path)
                print("New best model (binary) !")
                return metrics
            if CLASSIFICATION == 'cat':
                best_model_path = f"{CHEMIN_CAT}/best_model.h5"
                model.save(best_model_path)
                print("New best model (multiclass) !")
                return metrics

            #else :
            #    print('The new model is not better than the best model, try again ! :(')
            #    return best_metrics

        elif MODEL_TARGET == 'mlflow':
            best_model = load_model()
            if best_model == None:
                if CLASSIFICATION == 'binary':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("First model is saved as best model !")
                    pass
                if CLASSIFICATION == 'cat':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("First model is saved as best model !")
                    pass

            metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)
            print(f'new metrics are : {metrics}')

            best_metrics = evaluate_model(best_model, X_test, y_test,threshold=THRESHOLD)
            print(f'ancient metrics are : {best_metrics}')

            keys_ =list(metrics.keys())

            if metrics[keys_[2]]>best_metrics['Recall'] and metrics[keys_[1]]>0.5:
                if CLASSIFICATION == 'binary':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("New best model (binary) !")
                    return metrics
                if CLASSIFICATION == 'cat':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("New best model (multiclass) !")
                    return metrics

            else:
                print("Your model is less efficient than the last one :(")

        pass

    elif METADATA == 'yes':
        X_train_pixel, X_test_pixel, X_train_cat, X_test_cat, y_train_pixel, y_test_pixel, y_train_cat, y_test_cat = preprocess()
        y_train_pixel = tf.cast(y_train_pixel, dtype=tf.float32)

        print (y_train_pixel)
        model = initialize_model()
        print('model cnn initialized...')

        model = compile_model(model)
        print('model cnn compiled')

        model, history = train_model(model, X_train_pixel,y_train_pixel)
        print('model cnn trained!')

        model_ml = initialize_fit_model_ml(X_train_cat,y_train_cat)
        print('model ml trained !')

        if MODEL_TARGET == 'local':
            #Load the best model
            best_model, best_model_ml = load_best_model()
            if best_model == None:
                if CLASSIFICATION == 'binary':
                    best_model_path = f"{CHEMIN_META_BINARY}/best_model.h5"
                    model.save(best_model_path)
                    print("First model cnn is saved as best model !")

                    best_model_ml_path = f"{CHEMIN_META_BINARY}/best_model_ml.h5"
                    model.save(best_model_ml_path)
                    print("First model ml is saved as best model !")
                    pass
                if CLASSIFICATION == 'cat':
                    best_model_path = f"{CHEMIN_META_CAT}/best_model.h5"
                    model.save(best_model_path)
                    print("First model cnn is saved as best model !")

                    best_model_ml_path = f"{CHEMIN_META_CAT}/best_model_ml.h5"
                    model.save(best_model_ml_path)
                    print("First model ml is saved as best model !")
                    pass

            best_accuracy, best_recall = evaluate_model(best_model,X_test_pixel,y_test_pixel,threshold=THRESHOLD, batch_size=256,
                                          model_ml=best_model_ml, X_test_cat=X_test_cat,
                                          X_test_pixel=X_test_pixel, weight_cnn=0.9)
            print(f'ancient metrics are : {best_accuracy, best_recall}')

            accuracy, recall = evaluate_model(model,X_test_pixel,y_test_pixel,threshold=THRESHOLD, batch_size=256,
                                          model_ml=model_ml, X_test_cat=X_test_cat,
                                          X_test_pixel=X_test_pixel, weight_cnn=0.9)
            print(f'new metrics are : {accuracy, recall}')

            #if best_recall>recall and accuracy>0.5:
                #if CLASSIFICATION == 'binary':
                 #   best_model_path = f"{CHEMIN_META_BINARY}/best_model.h5"
                  #  model.save(best_model_path)
                   # print("New best model cnn (binary) !")

                    #best_model_ml_path = f"{CHEMIN_META_BINARY}/best_model_ml.h5"
                    #model.save(best_model_ml_path)
                    #print("First model ml is saved as best model !")

                    #return accuracy, recall

                #if CLASSIFICATION == 'cat':
                 #   best_model_path = f"{CHEMIN_META_CAT}/best_model.h5"
                  #  model.save(best_model_path)
                   # print("New best model cnn (multiclass) !")

                    #best_model_ml_path = f"{CHEMIN_META_CAT}/best_model_ml.h5"
                    #model.save(best_model_ml_path)
                    #print("First model ml is saved as best model !")
                    #return accuracy, recall

            #else :
             #   print('The new model is not better than the best model, try again ! :(')
             #   return best_accuracy, best_recall

        elif MODEL_TARGET == 'mlflow':
            best_model = load_model()
            if best_model == None:
                if CLASSIFICATION == 'binary':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("First model is saved as best model !")
                    pass
                if CLASSIFICATION == 'cat':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("First model is saved as best model !")
                    pass

            metrics = evaluate_model(model, X_test, y_test,threshold=THRESHOLD)
            print(f'new metrics are : {metrics}')

            best_metrics = evaluate_model(best_model, X_test, y_test,threshold=THRESHOLD)
            print(f'ancient metrics are : {best_metrics}')

            keys_ =list(metrics.keys())

            if metrics[keys_[2]]>best_metrics['Recall'] and metrics[keys_[1]]>0.5:
                if CLASSIFICATION == 'binary':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("New best model (binary) !")
                    return metrics
                if CLASSIFICATION == 'cat':
                    save_model(model)
                    mlflow_transition_model('None', 'Production')
                    print("New best model (multiclass) !")
                    return metrics

            else:
                print("Your model is less efficient than the last one :(")

        pass


if __name__ == '__main__':
    #preprocess()
    train()
