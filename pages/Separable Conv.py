#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:17:36 2023

@author: martin
"""

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from skimage import data
from tensorflow.keras import layers, models, Sequential, Model

@st.cache_data
def load_image():
    """
    

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    image = data.load('cat')
    image = np.expand_dims(image, axis=0)
    return image

@st.cache_data
def generate_kernel(filter_size):
    """
    

    Parameters
    ----------
    filter_size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pass

@st.cache_resource
def whole_conv_model(filter_size):
    """
    

    Parameters
    ----------
    filter_size : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

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
    

    Parameters
    ----------
    filter_size : TYPE
        DESCRIPTION.
    h_filters : TYPE
        DESCRIPTION.
    v_filters : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

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
    # plot loss
    pass

#%%

image = load_image()

with st.form('Experiment settings'):
    size = st.slider('Kernel size', value=3, max_value=31, min_value=3, step=2)
    h_filters = st.slider('Number of horizontal filters', value=1, 
                          max_value=size, min_value=1
                          )
    v_filters = st.slider('Number of vertical filters', value=1, 
                          max_value=size, min_value=1
                          )
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        kernel = generate_kernel(size)
        whole_model = whole_conv_model(size)
        sep_model = sep_conv_model(size, h_filters, v_filters)
        
        train_model(whole_model, image)
        train_model(sep_model, image)
        