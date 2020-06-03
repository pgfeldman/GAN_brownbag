import Influx2_ML.OneDGAN2 as ODG

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalMaxPool1D, MaxPool1D, UpSampling1D, Reshape, Flatten
import tensorflow.keras.backend as K
import math
from typing import List, Dict

class OneDGAN2a(ODG.OneDGAN2):
    def __init__(self):
        super().__init__()
        self.n_critic = 1

    # define the standalone discriminator model
    def define_discriminator(self) -> Sequential:
        # activation_func = tf.keras.layers.LeakyReLU(alpha=0.02)
        activation_func = tf.keras.layers.ReLU()
        self.d_model = Sequential()
        self.d_model.add(Conv1D(filters=self.vector_size, kernel_size=20, strides=4, activation=activation_func, batch_input_shape=(self.num_samples, self.vector_size, 1)))
        self.d_model.add(Dropout(.3))
        self.d_model.add(GlobalMaxPool1D())
        self.d_model.add(Flatten())
        self.d_model.add(Dense(self.vector_size/2, activation=activation_func, kernel_initializer='he_uniform', input_dim=self.vector_size))
        self.d_model.add(Dropout(.3))
        self.d_model.add(Dense(1, activation='sigmoid'))

        # compile model
        loss_func = tf.keras.losses.BinaryCrossentropy()
        opt_func = tf.keras.optimizers.RMSprop(lr=0.0005)
        self.d_model.compile(loss=loss_func, optimizer=opt_func, metrics=['accuracy'])
        return self.d_model

# exercise the class
def main():
    create_and_save_models = True
    load_and_use_models = not create_and_save_models

    directory = "models"
    prefix = "OneDGAN2a"

    skip_training = False
    latent_dimension = 16
    num_samples = 200
    vector_size = 32
    span = math.pi*2.0
    offset = 2.0
    side = 5
    epochs = 5000
    eval = 100

    odg = OneDGAN2a()
    odg.setup(latent_dimension, num_samples, vector_size, span, offset)

    if create_and_save_models:
        odg.create_and_save_models(epochs, eval, side, directory, prefix, skip_training)

    if load_and_use_models:
        odg.load_and_use_models(directory, prefix)


if __name__ == "__main__":
    main()