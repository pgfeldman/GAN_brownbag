# train a generative adversarial network on a one-dimensional function
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalMaxPool1D, MaxPool1D, UpSampling1D, Reshape, Flatten
from matplotlib import pyplot as plt
import math
import os
from typing import List, Dict


class OneDGAN2:
    """
    Base class that builds a GAN for producing time-series data that attempts to mimic math functions.
    Inhereting classes would normally override generate_real_samples(), define_generator, and define_discriminator.
    Loosly based on the example at https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

    ...
    Attributes
    ---------
    g_model: Sequential
        The generator Keras model
    d_model: Sequential
        The discriminator Keras model
    gan_model: Sequential
        The GAN, which connects the generator and discriminator
    latent_dimension:int
        The size of the noise array
    num_samples:int
        The number of sample time series that we create. In this example, they are based on the same function,
        but sampled at different offsets into the function (e.g. f(x + offset))
    vector_size:int
        The length of the series we're working with
    span:float
        The range the function covers (e.g. -2PI - PI)
    offset:float
        The maximum offset

    Methods
    -----------
    reset()
       Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    generate_real_samples(self, num_samples:int) -> [np.array, np.array]
        generate n real samples with class labels
    generate_latent_points(self, latent_dim:int, num_samples:int, span:float=1.0) -> np.array:
        create the matrix of latent points
    define_generator(self) -> Sequential:
        create the Keras model to generate fake data
    define_discriminator(self) -> Sequential:
        create the Keras model to discriminate between real and fake data
    define_gan(self) -> Sequential:
        define the combined generator and discriminator model. Training happens using this model
    generate_fake_samples(self, num_samples:int, span:float=4.0) -> [np.array, np.array]:
        generate a set of time series that will be evaluated against the 'real' data
    save_models(self, directory:str, prefix:str):
        saves the models in TF2.x format to a specified directory
    def load_models(self, directory:str, prefix:str):
        loads models in TF2.x format from a specified directory
    def setup(self, latent_dimension:int, num_samples:int, vector_size:int, span:float=4.0, offset:float=0):
        sets the class-global variables and creates the Keras models
    plot_triptych(self):
        creates a plot that shows the 'real' data, a sequence of latent space vectors, and the synthesized output of
        the generator
    evaluate(self, epoch:int, data_dict:Dict):
        evaluates the loss and accuracy to the discriminator and saves data out for plots
    train(self, n_epochs:int=1000, n_eval:int=100, n_batch:int=128) -> Dict:
        train the model for a specified number of epochs and batch size. Periocially evaluate the accuracy and loss
    """
    g_model: Sequential
    d_model: Sequential
    gan_model: Sequential
    latent_dimension:int
    num_samples:int
    vector_size:int
    span:float
    offset:float
    fake_class_value:float
    n_critic:int

    def __init__(self):
        self.reset()
    # resets all the variables. Needed to eliminate class cross-contamination of class-global variables with
    # multiple instances of this class
    def reset(self):
        print("OneDGAN2.reset()")
        self.g_model = None
        self.d_model = None
        self.gan_model = None
        self.latent_dimension = 16
        self.num_samples = 10
        self.vector_size = 100
        self.span = 4.0
        self.offset = 0
        self.fake_class_value = 0
        self.n_critic = 1

    # generate n real samples with class labels
    def generate_real_samples(self) -> [np.array, np.array]:
        X = np.ndarray(shape=(self.num_samples, self.vector_size))
        for i in range(self.num_samples):
            pos = 0
            step = 0
            offset = 0

            if self.vector_size > 1:
                step = self.span/(self.vector_size-1)
                pos = -self.span*0.5
                offset = np.random.standard_normal()*self.offset

            for j in range(self.vector_size):
                X[i][j] = math.sin(pos+offset) * 0.5
                # X[i][j] = np.random.standard_normal()*0.1 # Noise test
                pos += step
        X = X.reshape(self.num_samples, self.vector_size, 1)

        # generate class labels. A value of 1 = real
        y = np.ones((self.num_samples, 1))
        return X, y

    # create the matrix of latent points that has a sides equal to the dimensions
    def generate_latent_points(self, scalar:float=1.0) -> np.array:
        samples = int(self.num_samples * scalar)
        x_input = np.random.randn(samples * self.latent_dimension)*self.span
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(samples, self.latent_dimension)
        return x_input

    # create the Keras model to generate fake data (Dense to CNN)
    def define_generator(self) -> Sequential:
        self.g_model = Sequential()
        self.g_model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dimension))
        self.g_model.add(BatchNormalization())
        self.g_model.add(Dense(self.vector_size, activation='tanh'))
        self.g_model.add(Reshape((self.vector_size, 1)))
        print("g_model_Dense.output_shape = {}".format(self.g_model.output_shape))

        # compile model
        loss_func = tf.keras.losses.BinaryCrossentropy()
        opt_func = tf.keras.optimizers.Adam(0.001)
        self.g_model.compile(loss=loss_func, optimizer=opt_func)
        return self.g_model

    # define the standalone discriminator model
    def define_discriminator(self) -> Sequential:
        # activation_func = tf.keras.layers.LeakyReLU(alpha=0.02)
        activation_func = tf.keras.layers.ReLU()
        self.d_model = Sequential()
        self.d_model.add(Conv1D(filters=self.vector_size, kernel_size=20, strides=4, activation=activation_func, batch_input_shape=(self.num_samples, self.vector_size, 1)))
        self.d_model.add(Dropout(.3))
        self.d_model.add(GlobalMaxPool1D())
        self.d_model.add(Flatten())
        self.d_model.add(Dense(25, activation=activation_func, kernel_initializer='he_uniform', input_dim=self.vector_size))
        self.d_model.add(Dropout(.3))
        self.d_model.add(Dense(1, activation='sigmoid'))

        # compile model
        loss_func = tf.keras.losses.BinaryCrossentropy()
        opt_func = tf.keras.optimizers.Adam(0.001)
        self.d_model.compile(loss=loss_func, optimizer=opt_func, metrics=['accuracy'])
        return self.d_model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self) -> Sequential:
        # make weights in the discriminator not trainable
        self.d_model.trainable = False
        # connect them
        self.gan_model = Sequential()
        # add generator
        self.gan_model.add(self.g_model)
        # add the discriminator
        self.gan_model.add(self.d_model)

        # compile model
        loss_func = tf.keras.losses.BinaryCrossentropy()
        opt_func = tf.keras.optimizers.Adam(0.001)
        self.gan_model.compile(loss=loss_func, optimizer=opt_func)
        return self.gan_model

    # generate a set of time series that will be evaluated against the 'real' data
    def generate_fake_samples(self) -> [np.array, np.array]:
        # generate points in latent space
        x_input = self.generate_latent_points()
        # predict outputs
        # X = x_input
        X = self.g_model.predict(x_input)
        # create class labels, a value of 0 means fake
        y = np.ones((self.num_samples, 1))* self.fake_class_value
        return X, y

    # saves the models in TF2.x format to a specified directory
    def save_models(self, directory:str, prefix:str):
        p = os.getcwd()
        os.chdir(directory)
        self.d_model.save("{}_discriminator.tf".format(prefix))
        self.g_model.save("{}_generator.tf".format(prefix))
        os.chdir(p)

    # loads models in TF2.x format from a specified directory
    def load_models(self, directory:str, prefix:str):
        p = os.getcwd()
        os.chdir(directory)
        self.d_model = tf.keras.models.load_model("{}_discriminator.tf".format(prefix))
        self.g_model = tf.keras.models.load_model("{}_generator.tf".format(prefix))
        self.define_gan()
        os.chdir(p)

    # sets the class-global variables and creates the Keras models
    def setup(self, latent_dimension:int, num_samples:int, vector_size:int, span:float=4.0, offset:float=0):
        self.latent_dimension = latent_dimension
        self.num_samples = num_samples
        self.vector_size = vector_size
        self.span = span
        self.offset = offset

        # create the discriminator
        self.define_discriminator()
        self.define_generator()
        self.define_gan()

        tf.keras.utils.plot_model(self.gan_model, to_file="generator_model.png")
        self.g_model.summary()

        tf.keras.utils.plot_model(self.gan_model, to_file="discriminator_model.png")
        self.d_model.summary()

        tf.keras.utils.plot_model(self.gan_model, to_file="gan_model.png")
        self.gan_model.summary()

    # creates a plot that shows the 'real' data, a sequence of latent space vectors, and the synthesized output of
    # the generator
    def plot_triptych(self):
        data_array, label_array = self.generate_real_samples()
        data_array = data_array.reshape(self.num_samples, self.vector_size)
        #print("data_array = \n{}".format(data_array))
        #print("label_array = \n{}".format(label_array))
        fig = plt.figure()
        fig.set_figwidth(15)

        plt.subplots_adjust(left=.08, right= .95)
        plt.subplot(131)
        plt.plot(data_array.T, marker='o')
        plt.title("Real Data")

        lpoint_array = self.generate_latent_points()
        #plt.figure(2)
        plt.subplot(132)
        plt.imshow(lpoint_array, aspect='auto', cmap='magma')
        plt.title("Latent Space")
        plt.xlabel("Dimensions")
        plt.ylabel("Samples")

        data_array, label_array = self.generate_fake_samples()
        data_array = data_array.reshape(self.num_samples, self.vector_size)
        # plt.figure(3)
        plt.subplot(133)
        plt.plot(data_array.T, marker='o')
        plt.title("Fake Data")

    # evaluates the loss and accuracy to the discriminator and saves data out for plots
    def evaluate(self, epoch:int, data_dict:Dict):
        r_acc_list: List = data_dict["real_acc"]
        f_acc_list: List = data_dict["fake_acc"]
        r_loss_list: List = data_dict["real_loss"]
        f_loss_list: List = data_dict["fake_loss"]
        # prepare real samples
        x_real, y_real = self.generate_real_samples()
        # evaluate discriminator on real examples
        loss_real, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples()
        # evaluate discriminator on fake examples
        loss_fake, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print("Epoch {}-----------------------------------------------".format(epoch+1))
        print("real accuracy = {:.2f}%, fake accuracy = {:.2f}%".format(acc_real*100, acc_fake*100))
        print("real loss = {:.4f}%, fake loss = {:.4f}%".format(loss_real, loss_fake))
        r_acc_list.append(acc_real)
        f_acc_list.append(acc_fake)
        r_loss_list.append(loss_real)
        f_loss_list.append(loss_fake)

    def plot_intermediate(self, axs, epochs:int, cur_epoch:int, side:int):
        num_subplots = side*side
        step_size = epochs/num_subplots
        cur_subplot = int(cur_epoch/step_size)

        if cur_epoch%step_size == 0:
            title = "epoch {}".format(cur_epoch)
            data_array, label_array = self.generate_fake_samples()
            data_array = data_array.reshape(self.num_samples, self.vector_size)
            x = int(cur_subplot / side)
            y = cur_subplot % side
            ax = axs[x, y]
            ax.plot(data_array.T)
            ax.set_title(title)
            plt.draw()
            plt.pause(0.001)


    # train the model for a specified number of epochs and batch size. Periocially evaluate the accuracy and loss
    def train(self, n_epochs:int=1000, n_eval:int=100, side:int = 4) -> Dict:
        # set up figure for intermediate plots
        fig = plt.figure()
        axs = fig.subplots(side, side)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        plt.subplots_adjust(left=.06, right= .95, top=0.95, bottom=0.05, hspace=0.3)

        #set up the dictionary to store the histories
        d_dict = {"real_acc":[], "real_loss":[], "fake_acc":[], "fake_loss":[]}
        # manually enumerate epochs
        for epoch in range(n_epochs):
            for i in range(self.n_critic):
                # prepare real samples
                # x_real, y_real = self.generate_real_samples(num_samples=half_batch)
                x_real, y_real = self.generate_real_samples()
                # prepare fake examples
                # x_fake, y_fake = self.generate_fake_samples(num_samples=half_batch)
                x_fake, y_fake = self.generate_fake_samples()
                # update discriminator
                self.d_model.train_on_batch(x_real, y_real)
                self.d_model.train_on_batch(x_fake, y_fake)

            # prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points()
            # create inverted labels for the fake samples (why?)
            y_gan = np.ones((self.num_samples, 1))
            # update the generator via the discriminator's error
            self.gan_model.train_on_batch(x_gan, y_gan)
            if (epoch+1) % n_eval == 0:
                self.evaluate(epoch, d_dict)
            self.plot_intermediate(axs, n_epochs, epoch, side)
        return d_dict

    def create_and_save_models(self, epochs:int, eval:int, side:int, directory:str, prefix:str, skip_training:bool=False):
        if skip_training:
            self.plot_triptych()
            plt.show()
            return

        actual_epochs = int(epochs/(side*side)) * side*side
        actual_eval = int( (actual_epochs/eval) / (epochs/eval)*eval)
        d = self.train(actual_epochs, actual_eval, side=side)
        self.plot_triptych()
        self.save_models(directory, prefix)

        fig = plt.figure()
        fig.set_figwidth(10)
        plt.subplot(121)
        plt.title("Fake")
        plt.plot(d["fake_acc"], label="accuracy")
        plt.plot(d["fake_loss"], label="loss")
        plt.legend()

        plt.subplot(122)
        plt.title("Real")
        plt.plot(d["real_acc"], label="accuracy")
        plt.plot(d["real_loss"], label="loss")
        plt.legend()

        plt.show()

    def load_and_use_models(self, directory:str, prefix:str):
        d_dict = {"real_acc":[], "real_loss":[], "fake_acc":[], "fake_loss":[]}
        epoch = 1
        self.load_models(directory, prefix)
        # self.evaluate(epoch, data_dict=d_dict)
        self.plot_triptych()
        plt.show()

# exercise the class
def main():
    create_and_save_models = True
    load_and_use_models = not create_and_save_models

    directory = "models"
    prefix = "test3"

    skip_training = False
    latent_dimension = 16
    num_samples = 200
    vector_size = 32
    span = math.pi*2.0
    offset = 2.0
    side = 5
    epochs = 5000
    eval = 100

    odg = OneDGAN2()
    odg.setup(latent_dimension, num_samples, vector_size, span, offset)

    if create_and_save_models:
        odg.create_and_save_models(epochs, eval, side, directory, prefix, skip_training)

    if load_and_use_models:
        odg.load_and_use_models(directory, prefix)


if __name__ == "__main__":
    main()