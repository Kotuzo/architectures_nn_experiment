from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
import logging as lg


logger = lg.getLogger(__name__)
model_name = ''


def create_model(params):
    global model_name
    logger.info('using template %s', __name__)
    logger.info('creating model %s', model_name)
    _model = Sequential()

    # ConvLayer 1
    _model.add(Convolution2D(
        32, (3, 3), input_shape=(32, 32, 3),
        padding='same', activation='elu'))
    _model.add(BatchNormalization())

    # ConvLayer 2
    _model.add(Convolution2D(32, (3, 3), padding='same',
                             activation='elu'))
    _model.add((BatchNormalization()))
    _model.add(MaxPooling2D((2, 2)))
    _model.add(Dropout(0.2))

    # ConvLayer 3
    _model.add(Convolution2D(32, (3, 3), padding='same',
                             activation='elu'))
    _model.add((BatchNormalization()))

    # ConvLayer 4
    _model.add(Convolution2D(32, (3, 3), padding='same',
                             activation='elu'))
    _model.add((BatchNormalization()))
    _model.add(Dropout(0.2))

    #ConvLayer 5
    _model.add(Convolution2D(32, (3, 3), padding='same',
                             activation='elu'))
    _model.add((BatchNormalization()))
    _model.add(MaxPooling2D((2, 2)))

    _model.add(Flatten())

    # FCLayer 1
    _model.add(Dense(params[0], activation='elu'))
    _model.add(Dropout(0.25))
    _model.add(BatchNormalization())

    # FCLayer 2
    _model.add(Dense(params[1], activation='elu'))
    _model.add(BatchNormalization())

    # FCLayer 3
    _model.add(Dense(params[2], activation='elu'))
    _model.add(Dropout(0.5))
    _model.add(BatchNormalization())

    _model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-3)
    _model.compile(loss='categorical_crossentropy', optimizer=sgd,
                   metrics=['accuracy'])
    return _model