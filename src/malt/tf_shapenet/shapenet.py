# Classification and clustering of 2d-outlines (shapes) using tensorflow.
# Script based on the materials of the MIT Workshop "Kintsugi, Upcycling and
# Machine Learning", held in summer 2020.
#
# Workshop Materials & Examples by Daniel Marshall & Yijiang Huang
#
# For references see:
# https://architecture.mit.edu/subject/summer-2020-4181
# https://docs.google.com/document/d/1qO5-4QdBO_dp3kl_R9rK6IvAdyQ0pFmJr8JDxQLjWc4/edit?usp=sharing
# https://architecture.mit.edu/sites/architecture.mit.edu/files/attachments/course/20SU-4.181_syll_marshall%2Bmueller.pdf
#
# Adapted by Max Eschenbach, DDU, TU-Darmstadt (2021)

# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os

# THIRD PARTY IMPORTS ---------------------------------------------------------


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# PATH TO THIS DIRECTIORY -----------------------------------------------------

_HERE = os.path.dirname(__file__)


def initial_train(model_name: str,
                  train_input: np.array,
                  train_result: np.array,
                  test_input: np.array,
                  test_result: np.array,
                  epochs: int = 100,
                  overwrite: bool = False):
    """
    Create a tensorflow keras model and run initial training for a specified
    number of epochs.
    """
    # check if model exists and throw error if it does
    model_file = os.path.normpath(os.path.join(_HERE, model_name + ".h5"))
    if os.path.isfile(model_file) and not overwrite:
        raise RuntimeError("Model file already exists AND overwrite flag is "
                           "false! Use load_and_train if you want to train "
                           "further.")

    # create keras classifier model
    classifier = Sequential()

    # get dimensions from input data
    dims_in = train_input.shape[1]
    dims_out = train_result.shape[1]

    # add layers to the neural network model
    classifier.add(Dense(units=dims_in, activation="relu", input_dim=dims_in))
    classifier.add(Dense(units=100, activation="relu"))
    classifier.add(Dense(units=60, activation="relu"))
    classifier.add(Dense(units=32, activation="relu"))
    classifier.add(Dense(units=10, activation="relu"))
    classifier.add(Dense(units=dims_out, activation="relu"))

    # compile the network
    classifier.compile(optimizer="adam",
                       loss=tf.keras.losses.MeanAbsoluteError(),
                       metrics=["accuracy"])

    # fit the model to the training data for the specified number of epochs
    classifier.fit(train_input,
                   train_result,
                   batch_size=400,
                   epochs=epochs)

    prediction = classifier.predict(test_input)
    error = abs(test_result - prediction)
    print(error)

    # evaluate the model
    classifier.evaluate(test_input, test_result)

    # save the model to a .h5 file
    classifier.save(model_file, overwrite=overwrite, save_format="h5")

    return prediction


def load_and_train(model_name: str,
                   train_input: np.array,
                   train_result: np.array,
                   epochs: int = 1000):
    """
    Load an existing tensorflow keras model and train it further for the
    specified number of epochs.
    """
    # load the model with the specified name
    model_file = os.path.normpath(os.path.join(_HERE, model_name + ".h5"))
    model = tf.keras.models.load_model(model_file)
    # fit the model to the training data for the specified number of epochs
    model.fit(train_input,
              train_result,
              batch_size=400,
              epochs=epochs)
    # save the re-trained model
    model.save(model_file, overwrite=True, save_format="h5")


def forward_pass(data_input: np.array, model_name: str):
    """
    Load an existing tensorflow keras model and let it make a prediction based
    on the input data.
    """
    # load the model with the specified name
    model_file = os.path.normpath(os.path.join(_HERE, model_name + ".h5"))
    model = tf.keras.models.load_model(model_file)
    # let the model make a prediction based on the input data
    prediction = model.predict(data_input)
    # return the prediction result
    return prediction
