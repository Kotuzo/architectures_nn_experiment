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
from models.learn_interface import _loss_his, run_configurations

logger = lg.getLogger(__name__)


x_train = None
y_train = None
x_test = None
y_test = None
da_flag = True
num_classes = 10


def prepare_data():
    global x_train, x_test, y_train, y_test, da_flag
    logger.info("Preparing data")
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
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


def _init():
    if da_flag:
        prepare_data()


def _save_history(path):
    if not os.path.exists(path):
        os.makedirs(path)
    logger.info('dumping history to catalog: ' + path)
    with open(path + '/loss.dump', 'wb+') as fp:
        pickle.dump(_loss_his.losses, fp)
    with open(path + '/val_loss.dump', 'wb+') as fp:
        pickle.dump(_loss_his.val_losses, fp)
    with open(path + '/acc.dump', 'wb+') as fp:
        pickle.dump(_loss_his.acc, fp)
    with open(path + '/val_acc.dump', 'wb+') as fp:
        pickle.dump(_loss_his.val_acc, fp)


def run():
    _init()
    run_configurations(_save_history)
