#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:17:36 2023

@author: martin
"""

import os
import sys
import numpy as np
from scipy import signal
from skimage import data
from datetime import timedelta
from timeit import default_timer as timer
from tensorflow.keras import layers, models, optimizers


def make_conv_image(image, kernel, filter_size):
    # format kernel
    tensor = kernel[:, :, np.newaxis]
    tensor = np.repeat(tensor, 3, axis=2)
    tensor = np.expand_dims(tensor, -1)

    # define conv and set weights
    conv = layers.Conv2D(1, (filter_size, filter_size), activation='relu',
                         padding='same', use_bias=False, weights=[tensor])

    # count parameters
    conv.build((512, 512, 1))
    params = conv.count_params()

    # pass image through conv layer
    conv_image = conv(image)
    return conv_image, params


def my_conv2d(image, kernel):
    # Apply the convolution to each channel separately
    result = np.zeros_like(image)
    for i in range(image.shape[-1]):  # Loop over the channels
        result[:, :, i] = signal.convolve2d(
            image[:, :, i], kernel, mode='same'
        )
    return result


def gaussian_kernel(l=5, sigma=1.):
    """
    Generate a square Gaussian kernel

    Code taken from Stack Overflow answer
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Parameters
    ----------
    l : integer, optional
        Size of the side for a Gaussian kernel. The default is 5.
    sig : float, optional
        Sigma used to generate the Gaussian kernel. The default is 1..

    Returns
    -------
    Numpy array
        2D Gaussian kernel

    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def generate_kernel(n_kernels, filter_size):
    """
    Generate a kernel from a bunch of Gaussian kernels

    Parameters
    ----------
    n_kernels : integer
        Number of Gaussian kernels to generate.
    filter_size : integer
        Size of the generated filter.

    Returns
    -------
    f : 2D numpy array
        Kernel resulting from combination of Gaussian filters.

    """
    f = np.zeros((filter_size, filter_size))
    add = 1 if filter_size // 3 % 2 == 0 else 0
    upper = max(3, filter_size // 3 + add) + 1
    ksizes = np.arange(3, upper)

    for sigma in np.random.uniform(low=0.7, high=1., size=n_kernels):
        ksize = np.random.choice(ksizes)
        k = gaussian_kernel(ksize, sigma)
        krange = np.arange(filter_size - ksize)
        x = np.random.choice(krange)
        y = np.random.choice(krange)
        f[y:y+k.shape[0], x:x+k.shape[1]] += k

    return f


def sep_conv_model(filter_size, h_filters, v_filters):
    """
    Initialize a model with 1 layer of separate convolutions for height and
    width which should re-capitulate a more complex 2D kernel with less
    parameters

    Parameters
    ----------
    filter_size : integer
        Size of the filter in the model.
    h_filters : integer
        Number of horizontal filters.
    v_filters : integer
        Number of vertical filters.

    Returns
    -------
    model : Keras model
        Model with separate 1D convolutions

    """
    in_ = layers.Input(shape=(512, 512, 1))
    h = layers.Conv2D(h_filters, (1, filter_size), activation='relu',
                      padding='same', use_bias=False)(in_)
    v = layers.Conv2D(v_filters, (filter_size, 1), activation='relu',
                      padding='same', use_bias=False)(in_)
    concat = layers.Concatenate()([h, v])
    out = layers.Conv2D(1, 1, activation=None, padding='same',
                        use_bias=False)(concat)

    model = models.Model(inputs=in_, outputs=out)
    return model


def train_model(model, kernel, filter_size, savename, epochs=500):
    # prep training data
    image = data.camera().astype(np.float32)
    image = image / 255.
    image = np.expand_dims(image, axis=0)  # (1, 512, 512)
    image = np.expand_dims(image, axis=-1)  # (1, 512, 512, 1)
    conv_image, base_params = make_conv_image(image, kernel, filter_size)

    # set up training loop
    # reducelr = callbacks.ReduceLROnPlateau(
    #     'loss', patience=20, min_delta=0.05
    # )

    model.compile(optimizers.Adam(learning_rate=0.003), loss='mse')

    history = model.fit(
        image, conv_image, epochs=epochs, batch_size=1, verbose=2,
        # callbacks=[reducelr]
    )
    loss = history.history['loss']
    loss_savename = savename.replace('_model.h5', '_mse.txt')
    np.savetxt(loss_savename, np.array(loss))

    error = model.evaluate(image, conv_image)
    model.save(savename)

    return error, base_params


if __name__ == '__main__':
    # parameters for the experiment
    # filter size and # of Gaussians to use in the kernel
    s, n = [int(arg) for arg in sys.argv[1:]]

    ng = s * n
    savename = f'{s}x{s}_{ng}Gaussians.csv'
    k = generate_kernel(ng, s)

    # make experiment folder
    os.chdir('experiments')
    exp_fldr = savename.split('.')[0]
    if not os.path.exists(exp_fldr):
        os.mkdir(exp_fldr)
    os.chdir(exp_fldr)

    # determine which run this is
    base = f'{s}x{s}'
    this_run = sum([1 for fldr in os.listdir()]) + 1
    fldr = f'run{this_run:02}'
    os.mkdir(fldr)
    os.chdir(fldr)

    # save data
    np.save(savename.replace('.csv', '_kernel.npy'), k)

    with open(savename, "w") as f:
        f.write('S,NG,H,V,MSE,Params,BaseParams\n')

    start = timer()
    for h in range(1, s // 2 + 2):
        for v in range(1, s // 2 + 2):
            print(f'Working on {h}H and {v}V')
            model = sep_conv_model(s, h, v)
            params = model.count_params()
            modelname = f'{h}H-{v}V_model.h5'
            final_mse, base_params = train_model(model, k, s, modelname)
            with open(savename, "a") as f:
                f.write(
                    f'{s},{ng},{h},{v},{final_mse:3e},{params},{base_params}\n'
                )

    end = timer()
    print(f'Time elapsed: {timedelta(seconds=end-start)}')
