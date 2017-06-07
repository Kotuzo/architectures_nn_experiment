import models.model1 as m1
import models.model2 as m2
from monitor import LossHistory
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import time
import logging as lg
import data_ext
import os
import pickle


logger = lg.getLogger(__name__)


_model = None
_loss_his = None


batch_size = 32
num_classes = 10
epochs = 500
data_augmentation = True


configurations = {
    'model1': [],
    'model2': []
}


_base = 64
_multipliers = [4, 8, 16]


def init():
    '''
    temporary function - creates configurations of models
    '''
    global configurations
    for m in _multipliers[1:]:
        temp = int(_base * m)
        configurations['model1'].append(list([temp])*3)
    for m in _multipliers[:-2]:
        temp = int(_base * m)
        configurations['model1'].append(list([temp*4, temp*2, temp]))
    for m in _multipliers:
        temp = int(_base * m)
        configurations['model2'].append([temp])
    logger.info('learning module initialized')


def add_conf(model, param_list):
    configurations[model].append(param_list)


def init_model():
    global _model, _loss_his
    _model = None
    _loss_his = None


def fit_model():
    global _model, _loss_his
    logger.info('fit model...')
    _loss_his = LossHistory()
    es = EarlyStopping(patience=10)
    datagen = ImageDataGenerator()
    history = _model.fit_generator(datagen.flow(data_ext.x_train, data_ext.y_train,
                                   batch_size=batch_size,
                                   shuffle=True),
                                   steps_per_epoch=data_ext.x_train.shape[0] // batch_size,
                                   epochs=epochs,
                                   validation_data=(data_ext.x_test, data_ext.y_test),
                                   callbacks=[_loss_his, es]
                                   )


def get_model_name(arg_list):
    name = ''
    temp = [str(int(a)) for a in arg_list]
    if len(arg_list) == 3:
        name += 'M1-5(32,3)-ELU' + '-'.join(temp)
        m1.model_name = name
    elif len(arg_list) == 1:
        name += 'M2-5(32,3)-ELU' + '-'.join(temp)
        m2.model_name = name
    return name


def _choose_model(arg, params):
    args = {
        'M1' : m1.create_model,
        'M2' : m2.create_model
    }
    return args[arg](params)


def count_time(func):
    t1 = time.time()
    func()
    t2 = time.time()
    logger.info('time elapsed %0.2f sec', t2-t1)


def run_configurations(callbacks=[]):
    global _model, _loss_his
    init()
    for conf in configurations.keys():
        for params in configurations[conf]:
            name = get_model_name(params)
            init_model()
            _model = _choose_model(name[:2], params)
            count_time(fit_model)
            for c in callbacks:
                c('../history/' + name)


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
    data_ext._init()
    run_configurations([_save_history])