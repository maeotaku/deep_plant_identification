from common import *

import xml.etree.cElementTree as ET
import csv
import glob


def change_xml_fields(file_path, field_names, hierarchy):
    try:
        changed = False
        tree = ET.ElementTree(file=file_path)
        species = get_first_words(tree.findall("Species")[0].text, 2)
        for field_name in field_names:
            for node in tree.findall(field_name):
                if not node is None:
                    node.text = hierarchy[field_name][species]
                    changed = True
        if changed:
            tree.write(file_path)

    except Exception as e:
        print("Cannot read xml file", file_path, e)
        return None


def fill_missing_info(start_path, hierarchy):
    cont = 0
    for root, dirs, files in os.walk(start_path):
        for name in files:
            filename = os.path.join(root, name)
            standalone_name, extension = os.path.splitext(filename)
            if cont % 2000 == 0:
                print("Progress", cont)
            if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                xml_path = os.path.join(start_path, standalone_name + ".xml")
                change_xml_fields(xml_path, ["Genus", "Family"], hierarchy)
            cont+=1


def get_first_words(name, k):
    return ' '.join(name.upper().split(' ')[0:k])



def get_hierarchy(cvs_path):
    hierarchy = { "Family": {}, "Genus": {}}
    with open(cvs_path, "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            species = row[2]
            hierarchy["Family"][species] = row[0]
            hierarchy["Genus"][species] = row[1]
            print(row)
    print(hierarchy)
    return hierarchy

#separate in train and test sets
def fill_missing(start_path, xml_path):
    hierarchy = get_hierarchy(xml_path)
    fill_missing_info(start_path, hierarchy)
    print("Dataset ready")

#ORIGINAL_PATH = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations/"
#NEW_PATH = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/PlantCLEF2015/test_separated/"
ORIGINAL_PATH = r"/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/delete/"
XML_PATH = r"/Users/maeotaku/OneDrive/PhD_Research/Experiments/CIRAD_INRIA/Hierarchical/SpeciesByDataset.csv"
fill_missing(ORIGINAL_PATH, XML_PATH)

#ORIGINAL_PATH = "/opt/data_plantclef/PlantCLEF/PlantCLEF2015Data/test/"
#NEW_PATH = "/opt/convnet_models/Datasets/PlantCLEF2015/test_separated/"
#separate_in_folders(ORIGINAL_PATH, NEW_PATH, field_name="Species")
