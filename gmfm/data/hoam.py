
import numpy as np
import pickle


def get_hoam_data(path):

    with open(path, "rb") as f:
        obj = pickle.load(f)

    data = obj['sols']
    mu = obj['mu']
    return data, mu
