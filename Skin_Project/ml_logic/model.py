import numpy as np

import tensorflow as tf
from keras import Model, Sequential, layers, optimizers, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import os
import pandas as pd
from Skin_Project.params import *

def initialize_dumb_model():
    model = Sequential()
    model.add(layers.Conv2D(16, (4,4), input_shape=(28, 28, 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def initialize_model():
    model = Sequential()

  # First convolutional layer
    model.add(layers.Conv2D(filters = 64, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)))

  # Max pooling layer
    model.add(layers.MaxPool2D(pool_size = (2,2)))
  # Second convolutional layer
    model.add(layers.Conv2D(filters = 64,
               kernel_size = (3,3),
               padding = "same",
               activation = "relu"))

     # Max pooling layer
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    # Third convolutional layer
    model.add(layers.Conv2D(filters = 32,
               kernel_size = (3,3),
               padding = "same",
               activation = "relu"))

    # Max pooling layer
    model.add(layers.MaxPool2D(pool_size = (2,2)))

    # Fourth convolutional layer
    model.add(layers.Conv2D(filters = 256,
               kernel_size = (3,3),
               padding = "same",
               activation = "relu"))

  # Max pooling layer
    model.add(layers.MaxPool2D(pool_size = (2,2)))


    # Flattening layer
    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(units = 128,
                activation = "relu"))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model



def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam()
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
        patience=10,
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


'''old evaluate
def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    return metrics'''

def evaluate_model(model,X_test,y_test,threshold):
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
