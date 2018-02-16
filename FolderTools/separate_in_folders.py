from common import *

#import xml
import xml.etree.cElementTree as ET
#from lxml import etree
#import unidecode
#import string
#import codecs
#from xml.parsers import expat


def get_xml_field_value(file_path, field_name):
    try:
        #with codecs.open(file_path, 'r', 'utf-8') as f:
        #    dat = f.read()
        #dat = unidecode.unidecode(dat)
        #dat = ''.join([str(char) for char in dat if char in string.printable])
        #tree = xml.etree.ElementTree.fromstring(dat.encode('utf-8'))

        parser = ET.XMLParser(encoding="UTF-8")
        tree = ET.parse(file_path, parser=parser)

        for node in tree.findall(field_name):
            if not node is None:
                return node.text
            else:
                return None
    except Exception as e:
        print("Cannot read xml file", file_path, e)
        return None


#most likely used with the Author tag, but can be anything else like a date
def build_file_dictionary(start_path, field_name):
    field_species_files = {}
    files = get_file_list(start_path)
    n = len(files)
    cont = 0
    for filename in files:
        standalone_name, extension = os.path.splitext(filename)
        if cont % 2000 == 0:
            print("Progress", cont)
        if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
            xml_path = os.path.join(start_path, standalone_name + ".xml")
            field_value = get_xml_field_value(xml_path, field_name)
            if not field_value is None:
                field_value = get_first_words(field_value, 2)
                img_path = os.path.join(start_path, filename)
                if field_value in field_species_files.keys():
                    field_species_files[field_value].append(img_path)
                    field_species_files[field_value].append(xml_path)
                else:
                    field_species_files[field_value] = [ img_path, xml_path ]
        cont+=1
    return field_species_files

def get_first_words(name, k):
    return ' '.join(name.upper().split(' ')[0:k])

#separate in train and test sets
def separate_in_folders(start_path, new_path, field_name="Species"):
    create_folder(new_path)
    field_species_files = build_file_dictionary(start_path, field_name)
    print(field_species_files)
    for species_name in field_species_files.keys():
        print(species_name)
        output_folder = os.path.join(new_path, species_name)
        create_folder(output_folder)
        copy_files_simpler(output_folder, field_species_files[species_name])
    print("Dataset ready")

#ORIGINAL_PATH = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations/"
#NEW_PATH = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/test_separated/"
ORIGINAL_PATH = r"/home/gmorales/Descarga_imagenes/eol_package/data/"
NEW_PATH = r"/home/jcarranza/Datasets/RAW/PlantCLEF2018/"
separate_in_folders(ORIGINAL_PATH, NEW_PATH, field_name="Species")

#ORIGINAL_PATH = "/opt/data_plantclef/PlantCLEF/PlantCLEF2015Data/test/"
#NEW_PATH = "/opt/convnet_models/Datasets/PlantCLEF2015/test_separated/"
#separate_in_folders(ORIGINAL_PATH, NEW_PATH, field_name="Species")
