import pickle
import numpy as np


def save_pickle(thing, fn):
    with open("{}.pickle".format(fn), "wb") as handle:
        pickle.dump(thing, handle, protocol=3)


def load_pickle(fn):
    with open("{}.pickle".format(fn), "rb") as handle:
        thing = pickle.load(handle)
    return thing


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w
