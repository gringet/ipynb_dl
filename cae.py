import keras
import numpy as np
import os
import urllib.request

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Conv2D,Dense,Dropout,Flatten,Input,MaxPooling2D,Reshape,UpSampling2D
from keras.models import Model
from keras.activations import relu, linear, sigmoid, tanh

import talos as ta

# Constants
NB_INSTANCES = 5000
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
def model(images_train, labels_train, images_test, labels_test, params):
    input_img = Input(shape=(28,28,1))

    x = Conv2D(params['first_Conv2D_dim'], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # x = Dropout(0.5)(x)
    x = Conv2D(params['second_Conv2D_dim'], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()
	x = Dense(params['dense_dim'], activation='relu')(x)

    encoded = Dense(2, activation='linear')(x)

	x = Dense(params['dense_dim'], activation='relu')(x)
	
    x = Dense(49*params['second_Conv2D_dim'], activation='relu')(encoded)
    x = Reshape((7,7,params['second_Conv2D_dim']))(x)

    x = Conv2D(params['second_Conv2D_dim'], (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(params['first_Conv2D_dim'], (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)

    # autoencoder.load_weights(CHECKPOINTS_PATH + 'cae_model_adam_mse_30.hdf5')
    autoencoder.compile(optimizer='adam', 
	                    loss='mean_squared_error', 
						metrics=['accuracy'])

    # Train the model
    history = autoencoder.fit(images_train,images_train,
                              epochs=10,
                              batch_size=params['batch_size'],
                              shuffle=True,
                              verbose = 1,
                              validation_split=0.33
                              )
    
    return history, autoencoder

# Load dataset
# images_train, labels_train = load_quickdraw(DATA_PATH, DATA_CLASSES, NB_INSTANCES)
images_train, labels_train, images_test, labels_test = load_mnist()

params = {'first_Conv2D_dim':[8, 16, 32, 64],
		  'second_Conv2D_dim':[8, 16, 32, 64],
		  'dense_dim':[256, 512, 1024, 1536, 2048],
          'encoded_activation':[linear],
		  'batch_size':[200,400]}

t = ta.Scan(x=images_train,
            y=images_train,
            model=model,
            grid_downsample=1, 
            params=params,
            dataset_name='mnist',
            experiment_no='1',
            functional_model=True)