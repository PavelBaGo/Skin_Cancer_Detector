import numpy as np
import tensorflow as tf
from keras import Model, Sequential, layers, optimizers, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import os
import pandas as pd
from Skin_Project.params import *
from keras.utils import to_categorical
from sklearn.ensemble import AdaBoostClassifier

def initialize_model():
    model = Sequential()

    if CLASSIFICATION == 'cat':
        model = Sequential()
        model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                            strides=(4, 4), activation="relu",
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(7, activation="softmax"))
        return model

    if CLASSIFICATION == 'binary':
        model = Sequential()
        model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                            strides=(4, 4), activation="relu",
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(1, activation="sigmoid"))
        return model

def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam()

    if CLASSIFICATION=='cat':
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()
                               , tf.keras.metrics.FBetaScore(beta=2.0)])

    elif CLASSIFICATION=='binary':
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()
                               , tf.keras.metrics.FBetaScore(beta=2.0)])

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=30,
        validation_data=None, # overrides validation_split
        validation_split=0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = callbacks.EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    return model, history


def initialize_fit_model_ml(X_train_cat,y_train_cat):
    model_ml = AdaBoostClassifier()
    model_ml.fit(X_train_cat,y_train_cat)
    return model_ml

def avg_recall_accuracy(model_cnn, model_ml, X_test_cat, X_test_pixel, y_test, weight_cnn=0.5):
    y_pred_list=[]
    y_pred_cnn = model_cnn.predict(X_test_pixel)
    y_pred_gcb = model_ml.predict_proba(X_test_cat)
    for i in range(X_test_cat.shape[0]):
        y_pred_list.append(final_predict(y_pred_cnn[i,:],y_pred_gcb[i,:],weight_cnn))
    y_pred = to_categorical(y_pred_list, 7)
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, recall


def final_predict(y_pred_cnn, y_pred_ml, weight_cnn):
    y_pred = np.asarray(y_pred_cnn) * weight_cnn + np.asarray(y_pred_ml) * (1-weight_cnn)
    return np.argmax(y_pred)


def evaluate_model(model,X_test,y_test,threshold, batch_size=256,
                   model_ml=None, X_test_cat=None, X_test_pixel=None, weight_cnn=0.5):

    if model is None:
        print(f"No model to evaluate")
        return None

    if METADATA == 'no':
        if CLASSIFICATION == 'binary':
            y_pred = model.predict(X_test)
            threshold = THRESHOLD
            y_binary_predictions = (y_pred > threshold).astype(int)
            accuracy = accuracy_score(y_test, y_binary_predictions)
            precision = precision_score(y_test, y_binary_predictions)
            recall = recall_score(y_test, y_binary_predictions)
            f2 = fbeta_score(y_test, y_binary_predictions,beta = 2.0)

            metrics_dict = {'Threshold':threshold,'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F2 Score':f2}
            df_metrics = pd.DataFrame(metrics_dict,index=[threshold])
            return df_metrics

        if CLASSIFICATION == 'cat':

            metrics = model.evaluate(
                x=X_test,
                y=y_test,
                batch_size=batch_size,
                verbose=0,
                # callbacks=None,
                return_dict=True)

            return metrics
    elif METADATA == 'yes':
        return avg_recall_accuracy(model, model_ml, X_test_cat, X_test_pixel, y_test, weight_cnn=weight_cnn)

    else :
        print('Metadata?')
        pass
