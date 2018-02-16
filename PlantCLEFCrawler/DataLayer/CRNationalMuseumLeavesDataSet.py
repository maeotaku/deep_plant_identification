from Base.IPlantDataSet import *
from Base.TaggedImage import *
import csv
import os
import urllib
import uuid
import datetime

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

def get_file_list(a_dir):
    files = next(os.walk(a_dir))[2]
    return clean_hidden_files(files)

class CRNationalMuseumLeavesDataSet(IPlantDataSet):
    LABEL_EXTENSION = '.JPG'

    def __init__(self, destination_path, images_file_path):
        super(CRNationalMuseumLeavesDataSet, self).__init__(destination_path)
        #IPlantDataSet.__init__(self, destination_path)
        self.images_file_path = images_file_path
        print("Loading images files...")
        self.images_data, self.high = self._read_all_files(images_file_path)
        self.current = 0

    def _read_all_files(self, root):
        images_data = {}
        image_count = 0
        print("Root: " + root)
        for species_name in get_immediate_subdirectories(root):
            species_folder = os.path.join(root, species_name)
            for filename in get_file_list(species_folder):
                name, extension = os.path.splitext(filename)
                if extension.upper() == self.LABEL_EXTENSION:
                    file_path = os.path.join(species_folder, filename)
                    images_data[file_path] = species_name
                    image_count+=1
        return images_data, image_count

    def create_new_img(self, img_name, properties, local_filename):
        return TaggedImageLocalFile(img_name, properties=properties, local_filename=local_filename)

    def next(self):
        if self.current > self.high:
            raise StopIteration
        else:
            while self.current < self.high:
                self.current += 1
                keys = self.images_data.keys()
                file_path = keys[self.current - 1]
                species_name = self.images_data[file_path]
                properties = { "Species" : species_name }
                return  self.create_new_img(get_new_uuid(), properties, file_path)
            raise StopIteration
