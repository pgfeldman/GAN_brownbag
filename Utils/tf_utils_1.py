import tensorflow as tf
import os, shutil
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def setup_tensorboard(dir_str: str, windows_slashes:bool = True) -> List:
    if windows_slashes:
        dir_str = dir_str.replace("/", "\\")
    try:
        shutil.rmtree(dir_str)
    except:
        print("no file {} at {}".format(dir_str, os.getcwd()))

    # use TensorBoard, princess Aurora!
    # callbacks = [tf.keras.callbacks.TensorBoard(log_dir=dir_str, profile_batch = '500,510')]
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=dir_str, profile_batch=1)]
    return callbacks

def plot(history, type_str:str, fig_num:int = 1, show:bool = False):
    plt.figure(fig_num)
    plt.plot(history[type_str])
    plt.plot(history['val_{}'.format(type_str)])
    plt.title("model {}".format(type_str))
    plt.ylabel(type_str)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    if show:
        plt.show()

def plot_layers(feature, side:int = 3, show:bool = False):
    farray = np.array(feature[0])
    # print("{}".format(farray[:, :, 0]))

    num_images = side*side
    fig, axs = plt.subplots(side, side)
    for d in range(num_images):
        x = int(d/side)
        y = d%side
        axs[x, y].imshow(farray[:, :, d], aspect='auto', cmap='magma')
    if show:
        plt.show()

def plot_grid(axs, X:List, Y:List, cur_image:int, side:int = 3, ct:str = "blue", scatter:bool = False):
    x = int(cur_image/side)
    y = cur_image%side
    if scatter:
        axs[x, y].scatter(X, Y, color=ct)
    else:
        axs[x, y].plot(X, Y, color=ct)
    plt.draw()
    plt.pause(0.001)