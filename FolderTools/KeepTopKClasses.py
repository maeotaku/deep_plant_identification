import os
import uuid
import sys
import threading
from shutil import copyfile, copytree
import PIL
from PIL import Image, ImageFile
from multiprocessing import Process, Queue


EXTENSION_ATTR = [".XML"]
EXTENSION_IMG = [".JPG"]

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

def get_file_list(a_dir):
    files = next(os.walk(a_dir))[2]
    return clean_hidden_files(files)

def create_class_dict(root_input):
    classes = {}
    for class_name in get_immediate_subdirectories(root_input):
        complete_class_input_path = os.path.join(root_input, class_name)
        files = get_file_list(complete_class_input_path)
        classes[class_name] = len(files)
    return classes


def convert(root_input, root_output, async=False, k=1000):
    print("Root: " + root_input)
    create_folder(root_output)
    classes = create_class_dict(root_input)

    topK = sorted(classes, key=classes.get, reverse=True)[:k]
    for class_name in topK:
        complete_class_input_path = os.path.join(root_input, class_name)
        complete_class_output_path = os.path.join(root_output, class_name)
        try:
            copytree(complete_class_input_path, complete_class_output_path)
        except OSError as e:
            print("Cannot copy the folder for class", class_name, e)


INPUT_FOLDER = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/ASUHerbariumCompressedGenuses/"
OUTPUT_FOLDER = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/ASUHerbariumCompressedGenusesK/"
#INPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEF/Output/"
#OUTPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEFCompressed/"
convert(INPUT_FOLDER, OUTPUT_FOLDER, k=255)
