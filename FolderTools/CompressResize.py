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

def crop_img(img, perc_v, perc_h):
    if perc_v==-1 or perc_h==-1:
        return img
    h = img.size[1]
    w = img.size[0]
    nh = int(h * perc_v)
    nw = int(w * perc_h)
    img = img.crop((nw, nh, w - nw, h - nh))
    return img

def save_img(img, output_path):
    lock.acquire()
    img.save(output_path, quality=95, optimize=True)
    lock.release()
    #img.close()

def crop_center_and_scale_image(image_in, new_width, new_height):
    im = Image.open(image_in)

    width, height = im.size
    min_size = -1
    if width <= height:
        min_size = width
        left = 0
        top = (height - min_size)/2
        right = min_size
        bottom = (height + min_size)/2
    else:
        min_size = height
        left = (width - min_size)/2
        top = 0
        right = (width + min_size)/2
        bottom = min_size

    im = im.crop((left, top, right, bottom))
    new_size = new_width, new_height
    im = im.resize(new_size, Image.ANTIALIAS)
    im.save(image_out)

def compress_img_file(img, base_width):
    if base_width==-1:
        return img
    w = img.size[0]
    if base_width > w:
        base_width = w
    w_percent = (base_width/float(w))
    h_size = int((float(img.size[1])*float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
    return img



def process_img(file_path, output_path, resize_width=-1, crop_perc_v=-1, crop_perc_h=-1):
    try:
        img = Image.open(file_path)
        #img = oimg.copy()
        #oimg.close() #yeah this looks weird but there is a bug in PIL so workaround it!
        img = compress_img_file(img, resize_width)
        img = crop_img(img, crop_perc_v, crop_perc_h)
        save_img(img, output_path)
    except Exception as e:
        print("Failed to save and/or resize/compress image file...", e)
        pass

def convert(root_input, root_output, base_width = -1, crop_perc_v=-1, crop_perc_h=-1, async=False):
    images_data = {}
    image_count = 0
    print("Root: " + root_input)
    create_folder(root_output)
    for class_name in get_immediate_subdirectories(root_input):
        #class_folder = os.path.join(root_input, class_name)
        complete_class_output_path = os.path.join(root_output, class_name)
        complete_class_input_path = os.path.join(root_input, class_name)
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
                if extension.upper() in EXTENSION_ATTR:
                    if async:
                        launch_thread(threads, copy_file, (complete_file_input, complete_file_output))
                    else:
                        copy_file(complete_file_input, complete_file_output)
                elif extension.upper() in EXTENSION_IMG:
                    if async:
                        launch_thread(threads, process_img, (complete_file_input, complete_file_output, base_width, crop_perc_v, crop_perc_h))
                    else:
                        process_img(complete_file_input, complete_file_output, base_width, crop_perc_v, crop_perc_h)
            image_count+=1

INPUT_FOLDER = r"/data/carranza_expe/Datasets/Herbaria_PlantCLEF_Matches_Typos"
#r"/Volumes/VERBATIM HD 1/PlantCLEFStandard/Herbaria_Matches_PlantCLEF"
OUTPUT_FOLDER =  r"/data/carranza_expe/Datasets/Herbaria_PlantCLEF_Matches_Cropped"
#r"/Users/maeotaku/Documents/DatasetsNon1/TESTCROP"
#INPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEF/Output/"
#OUTPUT_FOLDER = "/home/jcarranza/code/PlantCLEFCrawler/Config/Herbaria_Matches_PlantCLEFCompressed/"
convert(INPUT_FOLDER, OUTPUT_FOLDER, base_width=-1, async=False, crop_perc_v=0.20, crop_perc_h=0.10)
#1024
