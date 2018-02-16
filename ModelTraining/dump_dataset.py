
import pickle
import numpy as np
import h5py
import json
import os

def open_hdf5_file(filename):
    f = h5py.File(filename, 'r')
    X = f["images"]
    y = f["labels"]
    return f, X, y

def dump_json(dict, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dict))

def dump_hierarchy(filename, outpath):
    h = open(filename, "rb")
    f = pickle.load(h)

    print(len(f['levels']['Species']))
    print(len(f['levels']['Genus']))
    print(len(f['levels']['Family']))

    dump_json(f['levels']['Species'], os.path.join(outpath, "Species.txt"))
    dump_json(f['levels']['Genus'], os.path.join(outpath, "Genus.txt"))
    dump_json(f['levels']['Family'], os.path.join(outpath, "Family.txt"))
    dump_json(f['Hierarchy'][1], os.path.join(outpath, "Genus_Species.txt"))
    dump_json(f['Hierarchy'][2], os.path.join(outpath, "Family_Species.txt"))
    dump_json(f['HierarchyInv'], os.path.join(outpath, "Species_Inv.txt"))

def dump_labels(filename, outpath):
    _, _, y = open_hdf5_file(filename)
    total = y.shape[0]
    print(y)

    with open(os.path.join(outpath,"Labels.txt"), 'w') as file:
        for i in range(0, total):
            file.write(str(y[i]) + "\n")

OUTPATH = r"/Datasets/HDF5/Herbaria1K/"
INPUT_PICKLE = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_hierarchies.pickle"
INPUT_HDF5 = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_train.hdf5"

dump_hierarchy(INPUT_PICKLE, OUTPATH)
dump_labels(INPUT_HDF5, OUTPATH)
