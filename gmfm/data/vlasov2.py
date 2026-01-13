
import numpy as np
import pickle

def get_vtwo_data():
    path = "/scratch/jmb1174/data_hoam/sde/vtwo.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)

    data = obj['sols']
    return data


def get_vbump_data():
    path = "/scratch/jmb1174/data_hoam/sde/vbump.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)

    data = obj['sols']
    return data
