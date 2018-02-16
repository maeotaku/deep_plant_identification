import numpy as np
import random
import caffe
import sys
import datetime
import lmdb
import json
from PIL import Image
import PIL
import os
import math
import shutil
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

from common import *

def get_xml_field_value(file_path, field_name=None):
    if field_name is None:
        return None
    try:
        tree = ET.ElementTree(file=file_path)
        for node in tree.findall(field_name):
            if node:
                return node.text
            else:
                return None
    except Exception as e:
        print("Cannot read xml file", file_path, e)
        return None

def get_xml_labels(file_path, field_names):
    try:
        tree = ET.ElementTree(file=file_path)
        attrs = []
        for field_name in field_names:
            for node in tree.findall(field_name):
                if not node is None:
                    attrs.append( node.text.upper() )
                else:
                    return None
        return attrs
    except Exception as e:
        print("Cannot read xml file", file_path, e)
        return None

def get_taxa_key(file_path, field_names):
    l = get_xml_labels(file_path, field_names)
    if not l is None:
        return tuple(l)
    else:
        return None

def clean_small_grouping(field_taxa_files, unknowns_field_taxa_files):
    for class_name in field_taxa_files:
        for field_value in field_taxa_files[class_name]:
            if len(field_taxa_files[class_name][field_value]) == 1:
                for filename in field_taxa_files[class_name][field_value]:
                    unknowns_field_taxa_files[class_name].append(filename)
                del field_taxa_files[class_name][field_value]

def make_file_info(img_path, xml_path):
    attrs = get_xml_labels(xml_path, ['Species', 'Genus', 'Family'])
    return [img_path] + attrs

def check_img_shape(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256), PIL.Image.ANTIALIAS)
    img = np.array(img)
    if not img.shape==(256,256,3):
        print("Weird", img_path)
        return None
    img = img[:,: ,::-1]
    img = np.transpose(img, (2,0,1))
    return img


#most likely used with the Author tag, but can be anything else like a date
def build_file_dictionary(start_path, field_name=None):
    field_taxa_files = {}
    unknowns_field_taxa_files = {}
    #class_name normally = species_name here]
    for class_name in get_immediate_subdirectories(start_path):
        species_folder = os.path.join(start_path, class_name)
        for filename in get_file_list(species_folder):
            standalone_name, extension = os.path.splitext(filename)
            if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                xml_path = os.path.join(species_folder, standalone_name + ".XML")
                field_value = get_xml_field_value(xml_path, field_name)
                img_path = os.path.join(species_folder, filename)
                taxa_key = get_taxa_key(xml_path, ['Species', 'Genus', 'Family'])

                if field_value is None:
                    unknowns_field_taxa_files[taxa_key].append(img_path)
                else:
                    if field_value in field_taxa_files[taxa_key]:
                        field_taxa_files[taxa_key][field_value].append(img_path)
                    else:
                        field_taxa_files[taxa_key][field_value] = [ img_path ]
    #some species+author will have only 1 image, for these cases we just want to add them generically to keep the proportion
    clean_small_grouping(field_taxa_files, unknowns_field_taxa_files)
    return field_taxa_files, unknowns_field_taxa_files

#most likely used with the Author tag, but can be anything else like a date
def build_file_dictionary_flat(start_path):
    unknowns_field_taxa_files = {}
    files = get_file_list(start_path)
    cont=0
    for filename in files[:10000]:
        standalone_name, extension = os.path.splitext(filename)
        if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
            img_path = os.path.join(start_path, filename)
            img_obj = check_img_shape(img_path)
            if not img_obj is None:
                xml_path = os.path.join(start_path, standalone_name + ".xml")
                taxa_key = get_taxa_key(xml_path, ['Species', 'Genus', 'Family'])
                if not taxa_key is None:
                    if taxa_key in unknowns_field_taxa_files.keys():
                        unknowns_field_taxa_files[taxa_key].append(img_obj)
                    else:
                        unknowns_field_taxa_files[taxa_key] = [ img_obj ]
            else:
                print("Weird image shape", img_path)
        if cont % 1000 == 0:
            print("Building file list...", cont, len(files))
        cont+=1
    return unknowns_field_taxa_files

def add_class(class_list, class_name):
    if not class_name in class_list:
        class_list.append(class_name)
    return class_list, class_list.index(class_name)

#separate in train and test sets
def separate_dataset(start_path, train_path, test_path, train_perc, field_name="Author"):
    create_folder(train_path)
    create_folder(test_path)
    field_taxa_files, unknowns_field_taxa_files = build_file_dictionary(start_path, field_name)

    class_taxa = {}
    species = []
    genus = []
    family = []

    train_files = []
    valid_files = []
    train_labels = []
    valid_labels = []
    for taxa_key in unknowns_field_taxa_files:
        current_species, current_genus, current_family = taxa_key
        species, s_idx = add_class(species, current_species)
        genus, g_idx = add_class(genus, current_genus)
        family, f_idx = add_class(family, current_family)

        current_files = unknowns_field_taxa_files[taxa_key]
        total = len(current_files)
        train_number = int(total * train_perc)
        test_number = total - train_number
        shuffle_all_files(current_files)

        train_files.append(current_files[0, train_number])
        valid_files.append(current_files[train_number, total])
        train_labels += [(s_idx, g_idx, f_idx)] * train_number
        valid_labels += [(s_idx, g_idx, f_idx)] * (total - train_number)

    for taxa_key in field_taxa_files:
        current_species, current_genus, current_family = taxa_key
        species, s_idx = add_class(species, current_species)
        genus, g_idx = add_class(genus, current_genus)
        family, f_idx = add_class(family, current_family)

        total = len(field_taxa_files[taxa_key].keys())
        train_number = int(total * train_perc)
        test_number = total - train_number

        specimen_cont = 0
        for specimen_idx in field_taxa_files[taxa_key]:
            if specimen_cont < train_number:
                train_files.append(current_files[0, train_number])
                train_labels += [(s_idx, g_idx, f_idx)] * train_number
            else:
                valid_files.append(current_files[train_number, total])
                valid_labels += [(s_idx, g_idx, f_idx)] * (total - train_number)
            specimen_cont+=1
    class_taxa["Species"] = species
    class_taxa["Genus"] = genus
    class_taxa["Family"] = family
    return train_files, train_labels, valid_files, valid_labels, class_taxa

def build_hierarchy_key(idx, names):
    return str(names.index(idx)) + "|" + idx

def add_node(hierarchy, parent, current):
    #if len(hierarchy.keys()) == 0:
    #    return
    for key in hierarchy.keys():
        if key == parent:
            if not current in hierarchy[parent].keys():
                hierarchy[parent][current] = {}
                return
        add_node(hierarchy[key], parent, current)

def create_taxonomy_dict(taxa_keys, class_taxa, file_path):
    #taxa_axis = { 0: "Species" : 1: "Genus", 2: "Family" }
    taxonomy = { "Names" : { "Family" : class_taxa["Family"], "Genus" : class_taxa["Genus"],  "Species" : class_taxa["Species"] } }
    root = "root"
    families = { root : {} }
    for taxa_key in taxa_keys:
        print(taxa_key)
        s_idx = taxa_key[0]
        s_key = build_hierarchy_key(s_idx, class_taxa["Species"])
        g_idx = taxa_key[1]
        g_key = build_hierarchy_key(g_idx, class_taxa["Genus"])
        f_idx = taxa_key[2]
        f_key = build_hierarchy_key(f_idx, class_taxa["Family"])

        add_node(families, root, f_key)
        add_node(families, f_key, g_key)
        add_node(families, g_key, s_key)

    taxonomy["Hierarchy"] = families
    with open(file_path, 'w') as fp:
        json.dump(taxonomy, fp)

def prepare_dataset(start_path, output_base, output_path, class_taxa=None):
    create_folder(output_path)
    unknowns_field_taxa_files = build_file_dictionary_flat(start_path)
    if class_taxa is None:
        class_taxa = {}
        species = []
        genus = []
        family = []
    else:
        species = class_taxa["Species"]
        genus = class_taxa["Genus"]
        family = class_taxa["Family"]

    files = []
    labels = []
    for taxa_key in unknowns_field_taxa_files:
        current_species, current_genus, current_family = taxa_key
        species, s_idx = add_class(species, current_species)
        genus, g_idx = add_class(genus, current_genus)
        family, f_idx = add_class(family, current_family)

        current_files = unknowns_field_taxa_files[taxa_key]
        total = len(current_files)
        shuffle_all_files(current_files)

        files += current_files
        labels += [ (s_idx, g_idx, f_idx), ] * total

    class_taxa["Species"] = species
    class_taxa["Genus"] = genus
    class_taxa["Family"] = family
    create_taxonomy_dict(unknowns_field_taxa_files.keys(), class_taxa, os.path.join(output_base, "Taxonomy.json"))
    #print(len(class_taxa["Species"]), len(class_taxa["Genus"]), len(class_taxa["Family"]))
    #print(len(files), len(labels))
    #files = files[:1000]
    #labels = labels[:1000]
    return files, labels, class_taxa

def create_specific_lmdb(path, images, labels):
    data_path = os.path.join(path, "data")
    label_path = os.path.join(path, "labels")

    #print(len(images), math.ceil(len(images)/1000.0))
    for idx in range(int(math.ceil(len(images)/1000.0))):

        in_db_data = lmdb.open(data_path, map_size=int(1e12))
        in_db_labels = lmdb.open(label_path, map_size=int(1e12))

        with in_db_data.begin(write=True) as in_txn_data:
            with in_db_labels.begin(write=True) as in_txn_labels:
                p_images = images[(1000*idx):(1000*(idx+1))]
                p_labels = labels[(1000*idx):(1000*(idx+1))]
                for in_idx in range(len(p_images)):
                    #img_path = p_images[in_idx]
                    label = p_labels[in_idx]
                    #img = Image.open(img_path)
                    #img = img.resize((256, 256), PIL.Image.ANTIALIAS)
                    #img = np.array(img)
                    '''
                    if len(img.shape) == 2:
                        print("Found no channels", img_path)
                        new_img = np.zeros(shape=(256, 256, 3))
                        new_img[:,:, 0] = img
                        new_img[:,:, 1] = img
                        new_img[:,:, 2] = img
                        img = new_img[:,: ,::-1]
                    if len(img.shape) == 3:
                        '''
                    #img = img[:,: ,::-1]
                    #img = np.transpose(img, (2,0,1))
                    img = p_images[in_idx]
                    img_dat = caffe.io.array_to_datum(img)
                    in_txn_data.put('{:0>10d}'.format(1000*idx + in_idx), img_dat.SerializeToString())
                    img_label = np.zeros((3,1,1), dtype=np.int)
                    img_label[0][0][0] = label[0]
                    img_label[1][0][0] = label[1]
                    img_label[2][0][0] = label[2]
                    #print(img_label)
                    img_label_dat = caffe.io.array_to_datum(img_label)
                    in_txn_labels.put('{:0>10d}'.format(1000*idx + in_idx), img_label_dat.SerializeToString())

        in_db_data.close()
        in_db_labels.close()

def create_and_separate_on_lmdb(start_path, train_path, test_path):
    train_files, train_labels, test_files, test_labels, class_taxa = separate_dataset(start_path, train_path, test_path, train_perc, field_name)
    assert len(train_files) == len(train_labels)
    assert len(valid_files) == len(valid_labels)
    create_specific_lmdb(train_path, train_files, train_labels)
    create_specific_lmdb(test_path, test_files, test_labels)


def create_lmdb(start_path, output_path, output_base, class_taxa=None):
    output_path = os.path.join(output_base, output_path)
    files, labels, class_taxa = prepare_dataset(start_path, output_base, output_path, class_taxa)
    assert len(files) == len(labels)
    create_specific_lmdb(output_path, files, labels)
    save_pickle(os.path.join(output_base, "Classes.pickle"), class_taxa)

OUTPUT_BASE = "/opt/convnet_models/Caffe/LMDB/PlantCLEF2015_Hiearchical/"
INPUT_PATH = "/opt/data_plantclef/PlantCLEF/PlantCLEF2015Data/train/"
OUTPUT_PATH = "train"
create_lmdb(INPUT_PATH, OUTPUT_PATH, OUTPUT_BASE)

classes_taxa = load_pickle(os.path.join(OUTPUT_BASE, "Classes.pickle"))
INPUT_PATH = "/opt/data_plantclef/PlantCLEF/PlantCLEF2015Data/test/"
OUTPUT_PATH = "test"
create_lmdb(INPUT_PATH, OUTPUT_PATH, OUTPUT_BASE, classes_taxa)
