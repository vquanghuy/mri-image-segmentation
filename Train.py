#!/usr/bin/env python
# coding: utf-8

# ## UNet Model

# In[2]:


import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import nibabel as nib
import math

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# import cv2
import tensorflow as tf

import tensorflow.keras.backend as tfback

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D,                     Activation, BatchNormalization, PReLU, Conv3DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from multiprocessing import freeze_support

def main():


    # In[3]:


    # get_ipython().system('nvidia-smi')


    # In[4]:

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


    # For GPU - use channels_first, otherwise use channel_last
    tfback.set_image_data_format("channels_first") # For GPU
    # tfback.set_image_data_format("channels_last") # For CPU


    # ## Build UNet model

    # In[5]:


    # Credit to: https://github.com/ellisdg/3DUnetCNN
    def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                                kernel=(3, 3, 3), activation=None,
                                padding='same', strides=(1, 1, 1),
                                instance_normalization=False):
        """
        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(
            input_layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)


    def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        deconvolution=False):
        if deconvolution:
            return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                strides=strides)
        else:
            return UpSampling3D(size=pool_size)

    def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                    pool_size=(2, 2, 2), n_labels=3,
                    initial_learning_rate=0.00001,
                    deconvolution=False, depth=4, n_base_filters=32,
                    include_label_wise_dice_coefficients=False, metrics=[],
                    batch_normalization=False, activation_name="sigmoid"):
        """
        Builds the 3D UNet Keras model.f
        :param metrics: List metrics to be calculated during model training (default is dice coefficient).
        :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
        coefficient for each label as metric.
        :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
        layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
        to train the model.
        :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
        layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
        divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
        increases the amount memory required during training.
        :return: Untrained 3D UNet Model
        """
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer,
                                            n_filters=n_base_filters * (
                                                    2 ** layer_depth),
                                            batch_normalization=batch_normalization)
            layer2 = create_convolution_block(input_layer=layer1,
                                            n_filters=n_base_filters * (
                                                    2 ** layer_depth) * 2,
                                            batch_normalization=batch_normalization)
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth - 2, -1, -1):
            up_convolution = get_up_convolution(pool_size=pool_size,
                                                deconvolution=deconvolution,
                                                n_filters=
                                                current_layer.shape[1])(
                current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
            current_layer = create_convolution_block(
                n_filters=levels[layer_depth][1].shape[1],
                input_layer=concat, batch_normalization=batch_normalization)
            current_layer = create_convolution_block(
                n_filters=levels[layer_depth][1].shape[1],
                input_layer=current_layer,
                batch_normalization=batch_normalization)

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)

        if not isinstance(metrics, list):
            metrics = [metrics]

        model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=loss_function,
                    metrics=metrics)
        return model


    # In[6]:


    def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                    epsilon=0.00001):
        dice_numerator = 2 * tfback.sum(y_true * y_pred, axis=axis) + epsilon
        dice_denominator = tfback.sum(y_true**2, axis=axis) + tfback.sum(y_pred**2, axis=axis) + epsilon
        dice_loss = 1 - tfback.mean(dice_numerator/dice_denominator)

        return dice_loss

    def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                        epsilon=0.00001):
        dice_numerator = 2 * tfback.sum(y_true * y_pred, axis=axis) + epsilon
        dice_denominator = tfback.sum(y_true, axis=axis) + tfback.sum(y_pred, axis=axis) + epsilon
        dice_coefficient = tfback.mean(dice_numerator/dice_denominator)
        
        return dice_coefficient


    # In[7]:


    model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient], depth=5)


    # In[8]:


    model.summary()


    # ## Train with Sequence

    # In[10]:


    class MSDSequence(Sequence):
        def __init__(self, sample_list, data_path, batch_size, sample_size):
            self.sample_list = sample_list
            self.data_path = data_path
            self.batch_size = batch_size
            self.sample_size = sample_size

        def __len__(self):
            return math.ceil(self.sample_size / self.batch_size)
        
        def __load_data(self, image_file_path, label_file_path):
            with open(image_file_path, 'rb') as f:
                X = np.load(f)
            with open(label_file_path, 'rb') as f:
                y = np.load(f)

            return X,y

        def __getitem__(self, idx):
            batch_list = self.sample_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = []
            batch_y = []
            
            for item in batch_list:
                X, y = self.__load_data(                 os.path.join(self.data_path, item['image']),                 os.path.join(self.data_path, item['label']),             )
                
                batch_x.append(X)
                batch_y.append(y)

            return np.array(batch_x), np.array(batch_y)


    # In[12]:


    DATA_DIR = "./Sample_Data"

    with open(os.path.join(DATA_DIR, "dataset.json")) as json_file:
        dataset = json.load(json_file)
        
    numTraining = dataset["numTraining"]
    trainingPropotion = math.ceil(0.8 * numTraining)

    trainingSet = dataset["training"][trainingPropotion:]
    validSet = dataset["training"][:trainingPropotion]
        
    train_generator = MSDSequence(trainingSet,                               DATA_DIR,                               batch_size=2,                               sample_size=len(trainingSet))
    valid_generator = MSDSequence(validSet,                               DATA_DIR,                               batch_size=2,                               sample_size=len(validSet))

    steps_per_epoch = 30
    n_epochs=50
    validation_steps = 30

    model.fit(train_generator, \
            steps_per_epoch=steps_per_epoch, \
            epochs=n_epochs, \
            use_multiprocessing=True, \
            validation_data=valid_generator, \
            validation_steps=validation_steps, \
            verbose=1)


# In[ ]:

if __name__ == '__main__':
    freeze_support()
    main()


