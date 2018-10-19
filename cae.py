import keras
import matplotlib.pyplot as plt
import numpy as np
import os
# import tensorflow as tf
import urllib.request

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Conv2D,Dense,Dropout,Flatten,Input,MaxPooling2D,Reshape,UpSampling2D
from keras.models import Model
from keras.activations import relu, linear

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

# Constants
NB_INSTANCES = 5000
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS,IMG_COLS,1)
BATCH_SIZE = 400
EPOCHS = 5
DATA_PATH = 'data/quickdraw/'
DATA_CLASSES = ['bat.npy','bird.npy','cake.npy','house.npy','monkey.npy','pizza.npy','saxophone.npy']
CHECKPOINTS_PATH = 'checkpoints/'
CHECKPOINTS_NAME = 'cae_model_adam_mae_10.hdf5'

# Functions definition
def load_quickdraw(data_path, data_classes, nb_instances):
    '''
        load quickdraw dataset from url
        ARGS:
            data_path: path in file system of data folder
            data_classes: names in file system of data files
            nb_instances: number of instances to retreive from each classe
        RETURNS:
            images_return: numpy array size nb_instance*nb_classesX28X28X1
            labels_return: numpy array size nb_instance*nb_classes of 0 to (nb_classes - 1) labels, depending on load order
    '''
    data_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    
    if not os.path.isdir(data_path):
        os.makedirs(data_path) 
    for i in range(len(data_classes)):
        if not os.path.isfile(data_path + data_classes[i]):
            urllib.request.urlretrieve(data_url + data_classes[i], data_path + data_classes[i])
            
    images_return = np.load(data_path + data_classes[0])[np.random.permutation(nb_instances)]
    labels_return = np.zeros(images_return.shape[0])
    for i in range(1, len(data_classes)):
        images_return = np.concatenate((images_return, np.load(data_path + data_classes[i])[np.random.permutation(nb_instances)]))
        labels_return = np.concatenate((labels_return, np.ones(images_return.shape[0] - labels_return.shape[0]) * i))
    return images_return.astype('float32') / 255, labels_return

def load_mnist():
    (images_train, labels_train) , (images_test,labels_test) = mnist.load_data()
    images_train = (images_train.astype('float32') / 255.).reshape((len(images_train),28, 28, 1))
    images_test = (images_test.astype('float32') / 255.).reshape((len(images_test),28, 28, 1))
    return images_train, labels_train, images_test, labels_test

def use_checkpoints(path, file_name):
    if not os.path.isdir(path):
        os.makedir(path)
    return ModelCheckpoint(path + file_name, monitor='loss', verbose=1, save_best_only=True, mode='auto')

# Create the model
def model(X_train, Y_train, X_test, Y_test):
    input_img = Input(shape=INPUT_SHAPE)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # x = Dropout(0.5)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)

    encoded = Dense(2, activation={{choice(['relu', 'sigmoid'])}})(x)

    x = Dense(392*2, activation='relu')(encoded)
    x = Reshape((7,7,16))(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)

    # autoencoder.load_weights(CHECKPOINTS_PATH + 'cae_model_adam_mse_30.hdf5')
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    autoencoder.fit(images_train,images_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    verbose = 1,
                    validation_split=0.33,
                    callbacks=[use_checkpoints(CHECKPOINTS_PATH, CHECKPOINTS_NAME)]
                   )
    
    score, acc = autoencoder.evaluate(X_test, Y_test, verbose = 1)
    return {'loss': -acc, 'status': STATUS_OK, 'encoder': encoder, 'autoencoder': autoencoder}

# Load dataset
# images_train, labels_train = load_quickdraw(DATA_PATH, DATA_CLASSES, NB_INSTANCES)
images_train, labels_train, images_test, labels_test = load_mnist()

best_run, best_model = optim.minimize(model=model,
                                      data=load_mnist,
                                      algo=tpe.suggest,
                                      max_evals=1,
                                      trials=Trials(),
                                      verbose=False)