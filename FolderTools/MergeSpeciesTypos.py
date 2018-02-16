import os
import uuid
import sys
import threading
from shutil import copyfile, copytree

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

def get_first_words(name, k):
    return ' '.join(name.split(' ')[0:k])

def copy_files(input_path, output_path):
    for filename in get_file_list(input_path):
        try:
            if not os.path.isfile(os.path.join(output_path, filename)):
                copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))
        except Exception as e:
            raise

def merge(root_input, root_output, async=False):
    print("Root: " + root_input)
    create_folder(root_output)
    for class_name in get_immediate_subdirectories(root_input):
        complete_class_input_path = os.path.join(root_input, class_name)
        normalized_class_name = get_first_words(class_name, 2).upper()
        print(normalized_class_name)
        complete_class_output_path = os.path.join(root_output, normalized_class_name)
        create_folder(complete_class_output_path)
        try:
            copy_files(complete_class_input_path, complete_class_output_path)
        except OSError as e:
            print("Cannot copy the folder for class", class_name, e)


INPUT_FOLDER = "/data/carranza_expe/Datasets/Herbaria_PlantCLEF_Matches_Compressed"
OUTPUT_FOLDER = "/data/carranza_expe/Datasets/Herbaria_PlantCLEF_Matches_Typos"
#INPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEF/Output/"
#OUTPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEFCompressed/"
merge(INPUT_FOLDER, OUTPUT_FOLDER)
