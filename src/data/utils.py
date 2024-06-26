import pickle
import json
import numpy as np

def load_pickle(filepath):
    with open(filepath, 'rb') as fh:
        data = pickle.load(fh)
    return data

def decompose_affine_mat(T):
    return T[0:3,0:3], T[0:3,3]