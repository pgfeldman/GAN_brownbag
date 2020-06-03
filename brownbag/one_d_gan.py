# example of generating random samples from X^2
import numpy as np
from matplotlib import pyplot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# generate randoms sample from x^2
# generate n real samples with class labels
def generate_real_samples(n = 100):
    # generate inputs in [-0.5, 0.5]
    X1 = np.random.rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    # generate class labels
    y = np.ones((n, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n) -> []:
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    return X


# define the standalone discriminator model
def define_discriminator(n_inputs=2) -> Sequential:
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_outputs=2) -> Sequential:
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model

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
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# train the discriminator model
def train_discriminator(model, n_epochs=1000, n_batch=128):
    half_batch = int(n_batch / 2)
    # run epochs manually
    flist = []
    rlist = []
    for i in range(n_epochs):
        # generate real examples
        X_real, y_real = generate_real_samples(half_batch)
        # update model
        model.train_on_batch(X_real, y_real)
        # generate fake examples
        X_fake, y_fake = generate_fake_samples(half_batch)
        # update model
        model.train_on_batch(X_fake, y_fake)
        # evaluate the model
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
        rlist.append(acc_real)
        flist.append(acc_fake)
        print("[{}] real = {:.2f}%, fake = {:.2f}%".format(i, acc_real*100, acc_fake*100))

    pyplot.plot(rlist, c="BLUE")
    pyplot.plot(flist, c="RED")
    pyplot.show()

def exercise_data_gen(generator:Sequential, latent_dim:int, n:int):
    # generate samples
    rdata, y = generate_real_samples()
    fdata = generate_fake_samples(generator, latent_dim, n)
    # plot samples
    pyplot.scatter(rdata[:, 0], rdata[:, 1], c="BLUE")
    pyplot.scatter(fdata[:, 0], fdata[:, 1], c="RED")
    pyplot.show()

def create_and_train_discriminator(n_epochs=500, n_batch=128) -> Sequential:
    # define the discriminator model
    model = define_discriminator()
    # summarize the model
    model.summary()
    # plot the model
    # plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

    # fit the model
    train_discriminator(model)
    return model

def create_generator(latent_dim:int) ->Sequential:
    # define the discriminator model
    model = define_generator(latent_dim)
    # summarize the model
    model.summary()
    return model


def main():
    latent_dim = 5
    d_model = define_discriminator()
    g_model = create_generator(latent_dim)
    exercise_data_gen(g_model, latent_dim, 100)

    gan_model = define_gan(d_model, g_model)
    gan_model.summary()




if __name__ == "__main__":
    main()

"""
# generate n fake samples with class labels
def generate_fake_samples(n = 100):
    # generate inputs in [-1, 1]
    X1 = -1 + np.random.rand(n) * 2
    # generate outputs in [-1, 1]
    X2 = -1 + np.random.rand(n) * 2
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2))
    # generate class labels
    y = np.zeros((n, 1))
    return X, y
"""