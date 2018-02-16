#!/usr/bin/python

# This work is licensed under the Creative Commons Attribution 3.0 United
# States License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by/3.0/us/ or send a letter to Creative
# Commons, 171 Second Street, Suite 300, San Francisco, California, 94105, USA.

# from http://oranlooney.com/make-css-sprites-python-image-library/
# Orignial Author Oran Looney <olooney@gmail.com>

#mods by Josh Gourneau <josh@gourneau.com> to make one big horizontal sprite JPG with no spaces between images
import os
import PIL
from PIL import Image
import glob
from common import *
import random






#most likely used with the Author tag, but can be anything else like a date
def build_file_dictionary(start_path, has_subfolders=True):
    field_species_files = []
    if has_subfolders:
        for species_name in get_immediate_subdirectories(start_path):
            species_folder = os.path.join(start_path, species_name)
            for filename in get_file_list(species_folder):
                standalone_name, extension = os.path.splitext(filename)
                if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                    img_path = os.path.join(species_folder, filename)
                    field_species_files.append(img_path)
    else:
        for filename in get_file_list(start_path):
            standalone_name, extension = os.path.splitext(filename)
            if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                img_path = os.path.join(start_path, filename)
                field_species_files.append(img_path)

    return field_species_files

def k_random_imgs(k, path, has_subfolder):
    '''
    iconMap = glob.glob(path)
    print(iconMap)
    return iconMap[0:k]
    '''
    images = build_file_dictionary(path, has_subfolder)
    random.shuffle(images)
    return images[0:k]


def create_sprite(path, rows, cols, w, h, has_subfolder=True):
    k = rows * cols
    #get your images using glob
    #just take the even ones
    iconMap = sorted(k_random_imgs(k, path, has_subfolder))
    #iconMap = iconMap[::2]

    print len(iconMap)

    images = [Image.open(filename) for filename in iconMap]

    print "%d images will be combined." % len(images)

    image_width, image_height = (w, h)#images[0].size

    print "all images assumed to be %d by %d." % (image_width, image_height)

    master_width = image_width * cols
    #seperate each image with lots of whitespace
    master_height = image_height * rows
    print "the master image will by %d by %d" % (master_width, master_height)
    print "creating image...",
    master = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0,0,0,0))  # fully transparent

    print "created."
    count=0
    for i in range(cols):
        #count, image in enumerate(images):
        col = image_width * i
        for j in range(rows):
            image = images[count]
            image = image.resize((image_width, image_height), PIL.Image.ANTIALIAS)
            row = image_height * j
            #print "adding %s at %d..." % (iconMap[count][1], col),
            master.paste(image,(col,row))
            count+=1
            print "added."
    print "done adding icons."

    print "saving master.jpg...",
    master.save('master.jpg', transparency=0 )
    print "saved!"

IMGS_PATH = r"/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations/"
#r"/Users/maeotaku/Documents/DatasetsNon1/CR_Noisy_Regions_Created/Calculated_EntireLeaf/"
#r"/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations/"
#r"/Volumes/VERBATIM HD 1/PlantCLEFStandard/Herbaria_Matches_PlantCLEF"
#r"/Users/maeotaku/Documents/Datasets/All_CR_Leaves_Cleaned/"
#r"/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/Herbaria_CR_Matches_Good_Separation_Compressed/"
IMG_WIDTH = 320
IMG_HEIGHT = 200

create_sprite(IMGS_PATH, 4, 6, IMG_WIDTH, IMG_HEIGHT, False)
