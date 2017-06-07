from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import logging as lg
from keras.optimizers import RMSprop, SGD
from monitor import LossHistory
import pickle
import keras
import os
import numpy as np
from random import shuffle


logger = lg.getLogger(__name__)


x_train = None
y_train = None
x_test = None
y_test = None
da_flag = True
num_classes = 10


def get_1000_samples_train(type):
    temp = []
    for (x, y) in (x_train, y_train):
        if y == type:
            temp.append(x)
            if len(temp) == 1000:
                return 


def get_200_samples_test(type):
    temp = []
    for (x, y) in (x_test, y_test):
        if y == type:
            temp.append(x)
            if len(temp) == 200:
                return temp


def create_dataset():
    global x_train, y_train, x_test, y_test

    temp_xtrain = []
    temp_ytrain = []
    temp_xtest = []
    temp_ytest = []

    for i in range(10):
        temp_xtrain.extend(get_1000_samples_train(i))
        temp_ytrain.extend(i)
        temp_xtest.extend(get_200_samples_test(i))
        temp_ytest.extend(i)

    x_train = np.array(temp_xtrain, dtype='uint8')
    y_train = np.array(temp_ytrain, dtype='uint8')
    x_test = np.array(temp_xtest, dtype='uint8')
    y_test = np.array(temp_ytest, dtype='uint8')


def shuffle_dataset():
    global x_train, y_train, x_test, y_test
    r_state = np.random.get_state()

    # shuffle train dataset
    np.random.shuffle(x_train)
    np.random.set_state(r_state)
    np.random.shuffle(y_train)

    # shuffle test dataset
    np.random.set_state(r_state)
    np.random.shuffle(x_test)
    np.random.set_state(r_state)
    np.random.shuffle(y_test)


def prepare_data(less_data=False):
    global x_train, x_test, y_train, y_test, da_flag
    logger.info("Preparing data")

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if less_data:
        create_dataset()
        shuffle_dataset()
    logger.info('x_train shape:' + str(x_train.shape))
    logger.info(str(x_train.shape[0]) + 'train samples')
    logger.info(str(x_test.shape[0]) + 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')   
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255
    da_flag = False
    # x_train = x_train[0:50]
    # y_train = y_train[0:50]
    # x_test = x_test[:25]
    # y_test = y_test[:25]


def _init():
    if da_flag:
        prepare_data()
