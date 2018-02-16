import os
import uuid
import sys
import threading
from shutil import copyfile
import PIL
from PIL import Image, ImageFile
from multiprocessing import Process, Queue


EXTENSION_ATTR = [".XML"]
EXTENSION_IMG = [".JPG"]

threads = []
MAX_THREADS = 2
lock = threading.Lock()
#queue = Queue()
#result = queue.get()
#print result

def launch_thread(threads, func, func_args):
    #t = threading.Thread(target=func, args=func_args)
    #t.start()
    p = Process(target=func, args=func_args)
    p.start()

    threads+= [ p ]
    if (len(threads) == MAX_THREADS):
        for p in threads:
            p.join()
        for p in threads:
            p.free()
        threads = []

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

def copy_file(file_path, output_path):
    try:
        copyfile(file_path, output_path)
    except Exception as e:
        print("Failed to copy file...", e)
        pass

def get_first_word(name):
    return name.split(' ', 1)[0].upper()

def convert(root_input, root_output, async=False):
    images_data = {}
    image_count = 0
    print("Root: " + root_input)
    create_folder(root_output)
    for class_name in get_immediate_subdirectories(root_input):
        complete_class_input_path = os.path.join(root_input, class_name)
        class_name = get_first_word(class_name)
        #class_folder = os.path.join(root_input, class_name)
        complete_class_output_path = os.path.join(root_output, class_name)
        create_folder(complete_class_output_path)
        for filename in get_file_list(complete_class_input_path):
            if image_count % 1000 == 0:
                print("Processing", image_count)
            complete_file_input = os.path.join(complete_class_input_path, filename)
            complete_file_output = os.path.join(complete_class_output_path, filename)
            name, extension = os.path.splitext(filename)
            #print(complete_file_output)
            #print(complete_file_input)
            if not os.path.exists(complete_file_output):
                if async:
                    launch_thread(threads, copy_file, (complete_file_input, complete_file_output))
                else:
                    copy_file(complete_file_input, complete_file_output)
            image_count+=1

INPUT_FOLDER = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/PlantCLEFStandard/ASUHerbariumCompressed/"
OUTPUT_FOLDER = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/PlantCLEFStandard/ASUHerbariumCompressedGenuses/"
#INPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEF/Output/"
#OUTPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEFCompressed/"
convert(INPUT_FOLDER, OUTPUT_FOLDER)
