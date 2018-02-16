import os
import re
import uuid
import urllib
import operator
import xml.etree.cElementTree as ET
from shutil import copyfile
from random import shuffle
import csv

from TaxaFilters import *

import pickle

def get_classes_idx(class_dict, class_name):
    if class_name.upper() in class_dict:
        idx = class_dict[class_name.upper()]
        return idx, class_dict
    else:
        idx = len(class_dict.keys())
        class_dict[class_name.upper()] = idx
        return idx, class_dict

def write_class_dict(classes_f, class_dict):
    if not classes_f is None:
        sorted_dict = sorted(class_dict.items(), key=operator.itemgetter(0))
        for class_name, idx in sorted_dict:
            write_list(classes_f, [idx, class_name])

def save_pickle(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def open_new_cvs(file_path):
    f = open(file_path, 'wb')
    writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    return f, writer

def append_to_cvs(f_pointer, writer, value_list):
    writer.writerow(value_list)

def close_cvs(f_pointer):
    f_pointer.close()

def filter_file_list_by_keywords(file_list, keyword_list=None):
    if keyword_list is None:
        return file_list
    print("Filtering file list")
    filtered_file_list = {}
    for folder in file_list.keys():
        upper_folder = folder.upper()
        for keyword in keyword_list:
            if keyword.upper() in upper_folder:
                filtered_file_list[folder] = file_list[folder]
                break;
    print("Filtered file list")
    return filtered_file_list

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

#def get_file_list(a_dir):
#    files = next(os.walk(a_dir))[2]
#    return clean_hidden_files(files)


def get_file_list(start_path, root_has_subfolders=False):
    print("Catching file list...")
    file_list = []
    if not root_has_subfolders:
        print("No subfolers...", start_path)
        for path, _, files in os.walk(start_path):
            for filename in files:
                #name, extension = os.path.splitext(filename)
                print("Adding ", filename)
                file_list.append(os.path.join(path, filename))
    else:
        print("Checking subfolders of", start_path)
        for species_name in get_immediate_subdirectories(start_path):
            print("Subfolder", species_name)
            species_folder = os.path.join(start_path, species_name)
            for filename in get_file_list(species_folder):
                #_, extension = os.path.splitext(filename)
                print("Adding ", filename)
                file_list.append(os.path.join(species_folder, filename))
    return file_list


def delete_files_with_pattern_name(start_path, pattern):
    for species_name in get_immediate_subdirectories(start_path):
        species_folder = os.path.join(start_path, species_name)
        for filename in get_file_list(species_folder):
            matchObj = re.match(pattern, filename)
            if not matchObj is None:
                os.remove(os.path.join(species_folder, filename))
    print("Files removed successfully.")

def shuffle_all_files(files):
    shuffle(files)

def open_file(filename, type_open="w"):
    return open(filename, type_open)

def write_list(f, line):
    f.write("{} {}\n".format(line[0], line[1]))

def close(f):
    f.close()

#Can be used to utilize the same class ids for different datasets when doing transfer learning
def caffe_class_file_to_dictionary(filename):
    caffe_classes = {}
    inv_caffe_classes = {}
    with open(filename,'r') as f:
        for line in f:
            idx, name = line.split()
            caffe_classes[name] = idx
            inv_caffe_classes[idx] = name
    return caffe_classes, inv_caffe_classes

def copy_files(path, current_files, idx, cont, max_number, file_pointer=None):
    while cont < max_number:
        filename = os.path.basename(current_files[cont])
        #print("Copying {} in {}\n".format(filename, os.path.join(path, filename)))
        copyfile(current_files[cont], os.path.join(path, filename))
        if not file_pointer is None:
            write_list(file_pointer, [filename, idx])
        cont+=1

def copy_files_simpler(output_path, current_files):
    for file_path in current_files:
        filename = os.path.basename(file_path)
        copyfile(file_path, os.path.join(output_path, filename))
