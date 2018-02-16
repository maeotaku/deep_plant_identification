from __future__ import print_function

import sys
import os
import time
import glob
import pickle
import h5py

def open_hdf5_file(filename):
    f = h5py.File(filename, 'r')
    X = f["images"]
    y = f["labels"]
    return f, X, y

def save_pickle(file_path, obj):
    if os.path.exists (file_path):
        file_path = file_path + "v2"
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

def get_file_list(a_dir):
    files = next(os.walk(a_dir))[2]
    return clean_hidden_files(files)
