import pickle


def open_pickle(name):
    with open(name, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        x, y = u.load()
        return x, y