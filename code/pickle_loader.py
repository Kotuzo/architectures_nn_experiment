import os
import pickle
import matplotlib.pyplot as plt

# plt.ion()
files = {
    'training acc': ['acc.dump', 'r'],
    'training loss': ['loss.dump', 'y'], 
    'validation acc': ['val_acc.dump', 'b'], 
    'validation loss': ['val_loss.dump','g']
}

# fig = plt.figure()

def _load_file(name):
    temp = pickle.load(open(name, 'rb'))
    return temp


def find_in_str(c, s):
    return [pos for pos, char in enumerate(s) if char == c]


def parse_dir(str):
    indexes = find_in_str('/', str)[-2:]
    return str[indexes[0]+1: indexes[1]]


def _add_suptitles(func):
    def wrapping(arg, limit, save=False):
        plt.close()
        fig = plt.figure()
        fig.canvas.set_window_title(parse_dir(arg))
        string = func(arg, limit)
        fig.suptitle(string)
        
        if save:
            plt.savefig('./'+parse_dir(arg)+'.png')
        plt.show()
    return wrapping


def _show_plot_acc(data, lbl, color):
    plt.subplot(211)
    plt.plot(data, color, label=lbl)
    plt.xlabel('epoch')
    plt.legend()


def _show_plot_loss(data, lbl, color):
    plt.subplot(212)
    plt.plot(data, color, label=lbl)
    plt.xlabel('epoch')
    plt.legend()


@_add_suptitles
def show_plots(model_dir, limit, save=False):
    flag = False
    for i,k in enumerate(files.keys()):
        f = model_dir + files[k][0]
        if not flag:
            log = 'epochs: {0}; '.format(len(_load_file(f)[:limit]))
            flag = True
        if 'acc' in k:
            _show_plot_acc(_load_file(f)[:limit], k, files[k][1])
            log += '{0}: {1:.2f} '.format(k, _load_file(f)[limit-1])
        else:
            _show_plot_loss(_load_file(f)[:limit], k, files[k][1])
            log += '{0}: {1:.2f} '.format(k, _load_file(f)[limit-1])
        if i % 3 == 0:
            log += '\n'
    return log


def sort_accuracies(model_dir, limit, metric):
    for i,k in enumerate(files.keys()):
        f = model_dir + files[k][0]
        if metric in k:
            acc = _load_file(f)[limit-1]
            name = parse_dir(model_dir)
    return name, acc


def run_all(dir, limit):
    for root, dir, files in os.walk(dir):
        if root is not dir:
            show_plots(root+'/', limit, True)


def pprint(tup_list):
    for el in tup_list:
        print(str(el[0]) + ' --- ' + str(el[1]))


def run_all_accuracy(dir, limit, metric='training acc'):
    temp = []
    for root, dir, files in os.walk(dir):
        if root is not dir and len(root) > 1:
            n, a = sort_accuracies(root+'/', limit, metric)
            temp.append((n,a))
    temp.sort(key=lambda tup: tup[1])
    pprint(temp)
