# train a generative adversarial network on a one-dimensional function
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import math
import Utils.tf_utils_1 as tfut

# define the standalone discriminator model
def define_discriminator(n_inputs=2) -> Sequential:
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim:int, n_outputs=2) -> Sequential:
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator:Sequential, discriminator:Sequential) -> Sequential:
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    loss_func = tf.keras.losses.BinaryCrossentropy()
    opt_func = tf.keras.optimizers.Adam(0.01)
    model.compile(loss=loss_func, optimizer=opt_func)
    return model

# generate n real samples with class labels
def generate_real_samples(num_samples:int, span:float=4.0):
    # generate inputs in [-span/2, span/2]
    step = span / num_samples
    X = np.ndarray(shape=(num_samples, 2))
    pos = -span/2.0
    for i in range(num_samples):
        X[i][0] = pos
        X[i][1] = math.sin(pos)
        pos += step

    # generate class labels
    y = np.ones((num_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim:int, num_samples:int, span:float=4.0) -> np.array:
    x_input = np.random.randn(latent_dim * num_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(num_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, num_samples:int, span:float=4.0) -> [np.array, np.array]:
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, num_samples, span)
    # predict outputs
    # X = x_input
    X = generator.predict(x_input)
    # create class labels
    y = np.zeros((num_samples, 1))
    return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(axs:plt.Axes, epoch:int, generator:Sequential, discriminator:Sequential,
                          latent_dim:int, cur_figure:int, side:int = 5, span:float = 4.0, num_samples:int=100) -> [float, float]:
    # prepare real samples
    x_real, y_real = generate_real_samples(num_samples, span=span)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, num_samples, span=span)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print("epoch = {}, real accuracy = {}%, fake accuracy = {}%".format(epoch, acc_real*100, acc_fake*100))
    # scatter plot real and fake data points
    tfut.plot_grid(axs, x_real[:, 0], x_real[:, 1], cur_figure, side=side, ct='red', scatter=True)
    tfut.plot_grid(axs, x_fake[:, 0], x_fake[:, 1], cur_figure, side=side, ct='blue', scatter=True)
    #plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    #plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    #plt.show()
    return(acc_real, acc_fake)

# train the generator and discriminator
def train(g_model:Sequential, d_model:Sequential, gan_model:Sequential, latent_dim:int, n_epochs:int=10000, n_batch:int=128, side:int = 5):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)

    fig_num = 0
    n_eval = n_epochs/(side*side)
    fig:plt.Figure
    axs:plt.Axes
    fig, axs = plt.subplots(side, side)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    span = math.pi * 2

    # manually enumerate epochs
    fake_list = []
    real_list = []
    for epoch in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch, span=span)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch, span=span)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch, span=span)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (epoch+1) % n_eval == 0:
            real, fake = summarize_performance(axs, epoch, g_model, d_model, latent_dim, fig_num, side, span)
            real_list.append(real)
            fake_list.append(fake)
            fig_num += 1
    plt.show()

    plt.plot(real_list, label="real accuracy", color="blue")
    plt.plot(fake_list, label="fake accuracy", color="red")
    plt.legend(loc="upper left")
    plt.show()


def train_model(latent_dim:int):
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(generator, discriminator)
    # train model
    train(generator, discriminator, gan_model, latent_dim, n_epochs=10000)


def exercise_parts(latent_dim:int):
    # create the generator
    generator = define_generator(latent_dim)
    batch = 128
    half_batch = int(batch/2)
    rX, ry = generate_real_samples(half_batch)
    fX, fy = generate_fake_samples(generator, latent_dim, half_batch)
    plt.scatter(rX[:, 0], rX[:, 1], color='red')
    plt.scatter(fX[:, 0], fX[:, 1], color='blue')
    plt.show()


def main():
    # size of the latent space
    latent_dim = 8
    # exercise_parts(latent_dim)
    train_model(latent_dim)



if __name__ == "__main__":
    main()