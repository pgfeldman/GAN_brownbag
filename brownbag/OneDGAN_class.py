# train a generative adversarial network on a one-dimensional function
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import math
import os
from typing import List, Dict

class OneDGAN:
    g_model: Sequential
    d_model: Sequential
    gan_model: Sequential
    latent_dimension:int
    num_samples:int
    vector_size:int
    span:float
    offset:float

    def __init__(self):
        self.reset()

    def reset(self):
        print("OneDGAN.reset()")
        self.g_model = None
        self.d_model = None
        self.gan_model = None
        self.latent_dimension = 16
        self.num_samples = 10
        self.vector_size = 100
        self.span = 4.0
        self.offset = 0

    # generate n real samples with class labels
    def generate_real_samples(self, num_samples:int) -> [np.array, np.array]:
        X = np.ndarray(shape=(num_samples, self.vector_size))
        for i in range(num_samples):
            pos = 0
            step = 0
            offset = 0

            if self.vector_size > 1:
                step = self.span/(self.vector_size-1)
                pos = -self.span*0.5
                offset = np.random.standard_normal()*self.offset

            for j in range(self.vector_size):
                X[i][j] = math.sin(pos+offset)
                pos += step

        # generate class labels. A value of 1 = real
        y = np.ones((num_samples, 1))
        return X, y

    def generate_latent_points(self, latent_dim:int, num_samples:int, span:float=1.0) -> np.array:
        x_input = np.random.randn(latent_dim * num_samples)*span
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(num_samples, latent_dim)
        return x_input

    def define_generator(self) -> Sequential:
        self.g_model = Sequential()
        self.g_model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dimension))
        self.g_model.add(Dense(self.vector_size, activation='tanh')) # activation was linear
        return self.g_model

    # define the standalone discriminator model
    def define_discriminator(self) -> Sequential:
        self.d_model = Sequential()
        self.d_model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.vector_size))
        self.d_model.add(Dense(1, activation='sigmoid'))
        # compile model
        self.d_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

    def generate_fake_samples(self, num_samples:int, span:float=4.0) -> [np.array, np.array]:
        # generate points in latent space
        x_input = self.generate_latent_points(self.latent_dimension, num_samples, span)
        # predict outputs
        # X = x_input
        X = self.g_model.predict(x_input)
        # create class labels, a value of 0 means fake
        y = np.zeros((num_samples, 1))
        return X, y

    def save_models(self, directory:str, prefix:str):
        p = os.getcwd()
        os.chdir(directory)
        self.d_model.save("{}_discriminator.tf".format(prefix))
        self.g_model.save("{}_generator.tf".format(prefix))
        os.chdir(p)

    def load_models(self, directory:str, prefix:str):
        p = os.getcwd()
        os.chdir(directory)
        self.d_model = tf.keras.models.load_model("{}_discriminator.tf".format(prefix))
        self.g_model = tf.keras.models.load_model("{}_generator.tf".format(prefix))
        self.define_gan()
        os.chdir(p)

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

    def plot_triptych(self):
        data_array, label_array = self.generate_real_samples(self.num_samples)
        #print("data_array = \n{}".format(data_array))
        #print("label_array = \n{}".format(label_array))
        fig = plt.figure()
        fig.set_figwidth(15)

        plt.subplots_adjust(left=.08, right= .95)
        plt.subplot(131)
        plt.plot(data_array.T, marker='o')
        plt.title("Real Data")

        lpoint_array = self.generate_latent_points(self.latent_dimension, self.latent_dimension)
        #plt.figure(2)
        plt.subplot(132)
        plt.imshow(lpoint_array, aspect='auto', cmap='magma')
        plt.title("Latent Space")
        plt.xlabel("Dimensions")
        plt.ylabel("Samples")

        data_array, label_array = self.generate_fake_samples(self.latent_dimension, self.num_samples)
        # plt.figure(3)
        plt.subplot(133)
        plt.plot(data_array.T, marker='o')
        plt.title("Fake Data")

    def evaluate(self, epoch:int, data_dict:Dict):
        r_acc_list: List = data_dict["real_acc"]
        f_acc_list: List = data_dict["fake_acc"]
        r_loss_list: List = data_dict["real_loss"]
        f_loss_list: List = data_dict["fake_loss"]
        # prepare real samples
        x_real, y_real = self.generate_real_samples(self.num_samples)
        # evaluate discriminator on real examples
        loss_real, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(self.num_samples)
        # evaluate discriminator on fake examples
        loss_fake, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print("Epoch {}-----------------------------------------------".format(epoch+1))
        print("real accuracy = {:.2f}%, fake accuracy = {:.2f}%".format(acc_real*100, acc_fake*100))
        print("real loss = {:.4f}, fake loss = {:.4f}%".format(loss_real, loss_fake))
        r_acc_list.append(acc_real)
        f_acc_list.append(acc_fake)
        r_loss_list.append(loss_real)
        f_loss_list.append(loss_fake)

    # train the generator and discriminator
    def train(self, n_epochs:int=1000, n_eval:int=100, n_batch:int=128) -> Dict:
        # determine half the size of one batch, for updating the discriminator
        half_batch = int(n_batch / 2)
        d_dict = {"real_acc":[], "real_loss":[], "fake_acc":[], "fake_loss":[]}

        # manually enumerate epochs
        for epoch in range(n_epochs):
            # prepare real samples
            x_real, y_real = self.generate_real_samples(num_samples=half_batch)
            # prepare fake examples
            x_fake, y_fake = self.generate_fake_samples(num_samples=half_batch)
            # update discriminator
            self.d_model.train_on_batch(x_real, y_real)
            self.d_model.train_on_batch(x_fake, y_fake)

            # prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points(self.latent_dimension, n_batch)
            # create inverted labels for the fake samples (why?)
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            self.gan_model.train_on_batch(x_gan, y_gan)
            if (epoch+1) % n_eval == 0:
                self.evaluate(epoch, d_dict)
        return d_dict


def main():
    create_and_save_models = True
    load_and_use_models = not create_and_save_models

    skip_training = False
    latent_dimension = 16
    num_samples = 20
    vector_size = 16
    span = 18.0
    offset = 2.0
    odg = OneDGAN()
    odg.setup(latent_dimension, num_samples, vector_size, span, offset)
    odg.plot_triptych()

    if create_and_save_models:
        if skip_training:
            plt.show()
            return

        batch_size = 128
        epochs = 10000
        eval = 100
        d = odg.train(epochs, eval, n_batch=batch_size)
        odg.plot_triptych()
        odg.save_models("models", "test1")

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

    if load_and_use_models:
        d_dict = {"real_acc":[], "real_loss":[], "fake_acc":[], "fake_loss":[]}
        epoch = 1
        odg.load_models("models", "test1")
        odg.evaluate(epoch, data_dict=d_dict)
        odg.plot_triptych()
        plt.show()

if __name__ == "__main__":
    main()