import logging as lg
import sys
from models.learn_interface import run 
import matplotlib.pyplot as plt
import os

os.environ['THEANO_FLAGS'] = 'device=gpu'

logger = lg.getLogger(__name__)
plt.ion()


def setup_logger():
    lg.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - \
        %(message)s', level=lg.DEBUG, filename='log.debug')
    console = lg.StreamHandler()
    console.setLevel(lg.INFO)
    formatter = lg.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    lg.getLogger('').addHandler(console)


def main(arg, specific=False):
    run()
    #if specific:
    #    logger.info('using specific model')
    #    data.run(int(arg))
    #else:
    #    logger.info('using range of models')
    #    for i in range(1, int(arg)+1):
    #        data.run(i)


if __name__ == '__main__':
    setup_logger()
    main('tralalala')
    #if len(sys.argv) == 2:
    #    main(sys.argv[1])
    #elif len(sys.argv) == 3 and sys.argv[2] == 'spec':
    #    main(sys.argv[1], True)
    #else:
    #    logger.critical('Bad number of arguments. Valid arguments: '
    #                    'integer - nb of models')