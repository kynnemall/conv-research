#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:17:36 2023

@author: martin
"""

import numpy as np
from skimage import data
from scipy import signal
from tensorflow.keras import layers, models, optimizers, callbacks


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
    in_ = layers.Input(shape=(300, 451, 3))
    h = layers.Conv2D(h_filters, (1, filter_size), activation='relu',
                      padding='same')(in_)
    h = layers.Conv1D(3, 1, activation='relu')(h)
    v = layers.Conv2D(v_filters, (filter_size, 1), activation='relu',
                      padding='same')(in_)
    v = layers.Conv1D(3, 1, activation='relu')(v)
    out = layers.Add()([h, v])

    model = models.Model(inputs=in_, outputs=out)
    return model


def train_model(model, kernel, epochs=300):
    # prep training data
    image = data.cat().astype(np.float32)
    image = image / 255.
    conv_image = my_conv2d(image, kernel)
    conv_image = np.expand_dims(conv_image, axis=0)
    image = np.expand_dims(image, axis=0)

    assert conv_image.shape == image.shape, f'{conv_image.shape}, {image.shape}'

    # set up training loop
    # reducelr = callbacks.ReduceLROnPlateau(
    #     'loss', patience=20, min_delta=0.05
    # )
    model.compile(optimizers.Adam(learning_rate=0.003), loss='mse')

    model.fit(
        image, conv_image, epochs=epochs, batch_size=1,
        # callbacks=[reducelr]
    )

    error = model.evaluate(image, conv_image)

    return error


s = int(input('Choose a filter size\n'))
n = int(
    input(
        'Choose a multiplier for the number of Gaussians between 2 and 4\n'
    )
)
ng = s * n
savename = f'experiments/{s}x{s}_{ng}Gaussians.csv'
k = generate_kernel(ng, s)
np.save(savename.replace('.csv', '_kernel.npy'), k)

with open(savename, "w") as f:
    f.write('S,NG,H,V,MSE,Params\n')

for h in range(1, s // 2 + 2):
    for v in range(1, s // 2 + 2):
        print(f'Working on {h}H and {v}V')
        model = sep_conv_model(s, h, v)
        params = model.count_params()
        final_mse = train_model(model, k)
        with open(savename, "a") as f:
            f.write(f'{s}, {ng}, {h}, {v}, {final_mse:.2f}, {params}\n')
