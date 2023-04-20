#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:17:36 2023

@author: martin
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from skimage import data
from tensorflow.keras import layers, models, optimizers, callbacks, Sequential


@st.cache_data
def load_image():
    """
    Load the 'cat' image from scikit-image

    Returns
    -------
    image : 2D numpy array
        Sample image.

    """
    image = data.load('cat')
    image = np.expand_dims(image, axis=0)
    return image


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


@st.cache_data
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


@st.cache_resource
def whole_conv_model(filter_size):
    """
    Initialize a model with 1 convolutional layer containing 1 filter

    Parameters
    ----------
    filter_size : integer
        Size of the filter in the model.

    Returns
    -------
    model : Keras model
        Single-layer Conv2D model

    """
    model = Sequential()
    model.add(
        layers.Conv2D(
            1, filter_size, activation='tanh', padding='same'
        )
    )
    return model


@st.cache_resource
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
    in_ = layers.Input(shape=())
    h = layers.Conv2D(h_filters, (1, filter_size), activation='tanh',
                      padding='same')(in_)
    v = layers.Conv2D(v_filters, (filter_size, 1), activation='tanh',
                      padding='same')(in_)
    out = layers.Add()([h, v])
    model = models.Model(inputs=in_, outputs=out)
    return model


def train_model(model, training_data):
    # set up training loop
    reducelr = callbacks.ReduceLROnPlateau(
        'loss', patience=10, min_delta=0.001
    )
    model.compile(optimizers.Adam(lr=0.003), loss='mse')

    history = model.fit(training_data, epochs=200, callbacks=[reducelr])

    # plot loss
    hist = pd.DataFrame(history)
    fig = px.line(data=hist, x='Epochs', y='Loss')

    return fig

# %%


image = load_image()

with st.form('Experiment settings'):
    size = st.slider('Kernel size', value=3, max_value=31, min_value=3, step=2)
    n_kernels = st.slider('Number of Gaussian filters to use in the kernel',
                          value=3, max_value=80, min_value=3, step=1)
    h_filters = st.slider('Number of horizontal filters', value=1,
                          max_value=size, min_value=1
                          )
    v_filters = st.slider('Number of vertical filters', value=1,
                          max_value=size, min_value=1
                          )

    submitted = st.form_submit_button("Submit")
    if submitted:
        kernel = generate_kernel(n_kernels, size)
        whole_model = whole_conv_model(size)
        sep_model = sep_conv_model(size, h_filters, v_filters)

        train_model(whole_model, image)
        train_model(sep_model, image)
