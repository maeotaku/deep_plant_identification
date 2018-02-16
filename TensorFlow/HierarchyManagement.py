from __future__ import print_function

import sys
import os
import numpy as np
import pickle

from Files import *

HIERARCHY_FILE_PARAM_LEVELS = "levels"
HIERARCHY_FILE_PARAM_HIERARCHY = "Hierarchy"
HIERARCHY_FILE_PARAM_HIERARCHY_INV = "HierarchyInv"

def build_level_mask(leaf_idx, level, hierarchy, inv_hierarchy, output_size):
    output = np.zeros((output_size), dtype='uint32')
    parent_idx = inv_hierarchy[leaf_idx][level]
    leaf_idxs = hierarchy[level][parent_idx]
    output[leaf_idxs] = 1
    #print(level, parent_idx, leaf_idx, leaf_idxs, np.sum(output), len(leaf_idxs))
    return output

def initialize_hierarchies(hierarchy_params_file, level_list, hierarchy_params=None):
    if hierarchy_params is None:
        classes_level_list = {}
        hierarchy = {}
        inv_hierarchy = {}
    else:
        print("Hierarchy file exists.")
        classes_level_list = hierarchy_params[HIERARCHY_FILE_PARAM_LEVELS]
        hierarchy = hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY]
        inv_hierarchy = hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY_INV]

    for level_label in level_list:
        print("Adding " + level_label + " in level list...")
        classes_level_list[level_label] = []

    print("Hierarchy Loaded.")
    print("Levels: ", level_list)
    print("Hierarchy: ", len(hierarchy.keys()))
    print("InvHierarchy: ", len(inv_hierarchy))
    return hierarchy_params, classes_level_list, hierarchy, inv_hierarchy

def initialize_hierarchies_from_file(hierarchy_params_file, level_list):
    if not os.path.isfile(hierarchy_params_file):
        print("File does not exists!"); classes_level_list = {}
        hierarchy = {}
        inv_hierarchy = {}
        for level_label in level_list:
            print("Adding " + level_label + " in level list...")
            classes_level_list[level_label] = []
    else:
        print("Hierarchy file exists.")
        hierarchy_params = load_pickle(hierarchy_params_file)
        classes_level_list = hierarchy_params[HIERARCHY_FILE_PARAM_LEVELS]
        hierarchy = hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY]
        inv_hierarchy = hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY_INV]

    print("Hierarchy Loaded.")
    print("Levels: ", level_list)
    print("Hierarchy: ", len(hierarchy.keys()))
    print("InvHierarchy: ", len(inv_hierarchy))
    return classes_level_list, hierarchy, inv_hierarchy

def save_hierarchies(hierarchy_params_file, hierarchy_params):
    #print(hierarchy_params)
    print("Savng hierarchy...")
    print(hierarchy_params)
    save_pickle(hierarchy_params_file, hierarchy_params)
    print("Done.")

def get_first_words(name, k):
    return ' '.join(name.split(' ')[0:k])

def add_class_on_level(classes_level_list, class_level, class_name, k=2):
    class_name = get_first_words(class_name, k).upper()
    if not class_name in classes_level_list[class_level]:
        classes_level_list[class_level] += [ class_name ]
    idx = classes_level_list[class_level].index( class_name )
    return idx

def add_to_hierarchy(hierarchy, inv_hierarchy, attrs):
    leaf_idx = attrs[0]
    level = 1 #0 is always the lowest
    while level < len(attrs):
        parent_idx = attrs[level]

        if level not in hierarchy:
            hierarchy[level] = {}

        if parent_idx not in hierarchy[level].keys():
            hierarchy[level][parent_idx] = [ leaf_idx ]
        else:
            if leaf_idx not in hierarchy[level][parent_idx]:
                hierarchy[level][parent_idx] += [ leaf_idx ]
        level+=1
    if leaf_idx not in inv_hierarchy:
        inv_hierarchy[leaf_idx] = attrs
