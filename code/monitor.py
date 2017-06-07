from keras.callbacks import Callback
import matplotlib.pyplot as plt
from collections import OrderedDict
import logging as lg
import time

logger = lg.getLogger(__name__)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.t1 = time.time()


    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

    def on_train_end(self, logs=None):
        logger.info('acc: %f; val_acc: %f; loss: %f; val_loss: %f', 
            self.acc[-1:][0], self.val_acc[-1:][0], 
            self.losses[-1:][0], self.val_losses[-1:][0])
        logger.info('model trained in %f minutes', (time.time() - self.t1)/60)


    def on_epoch_begin(self, epoch, logs=None):
        pass

