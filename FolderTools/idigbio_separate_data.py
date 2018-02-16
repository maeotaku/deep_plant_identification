from common import *

import xml.etree.cElementTree as ET

def get_xml_field_value(file_path, field_name):
    try:
        tree = ET.ElementTree(file=file_path)
        for node in tree.findall(field_name):
            if node:
                print("Found!", node.text)
                return node.text
            else:
                return None
    except Exception as e:
        print("Cannot read xml file", file_path, e)
        return None

def clean_small_grouping(field_species_files, unknowns_field_species_files):
    for species_name in field_species_files:
        for field_value in field_species_files[species_name]:
            if len(field_species_files[species_name][field_value]) == 1:
                for filename in field_species_files[species_name][field_value]:
                    unknowns_field_species_files[species_name].append(filename)
                del field_species_files[species_name][field_value]

#most likely used with the Author tag, but can be anything else like a date
def build_file_dictionary(start_path, field_name, keyword_list=None):
    field_species_files = {}
    unknowns_field_species_files = {}
    for species_name in get_immediate_subdirectories(start_path):
        if keyword_list is None or (not keyword_list is None and species_name in keyword_list ):
            field_species_files[species_name] = {}
            unknowns_field_species_files[species_name] = []
            species_folder = os.path.join(start_path, species_name)
            for filename in get_file_list(species_folder):
                standalone_name, extension = os.path.splitext(filename)
                if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                    xml_path = os.path.join(species_folder, standalone_name + ".xml")
                    field_value = get_xml_field_value(xml_path, field_name)
                    img_path = os.path.join(species_folder, filename)
                    if field_value is None:
                        unknowns_field_species_files[species_name].append(img_path)
                    else:
                        if field_value in field_species_files[species_name]:
                            field_species_files[species_name][field_value].append(img_path)
                        else:
                            field_species_files[species_name][field_value] = [ img_path ]
    #some species+author will have only 1 image, for these cases we just want to add them generically to keep the proportion
    clean_small_grouping(field_species_files, unknowns_field_species_files)
    return field_species_files, unknowns_field_species_files




#separate in train and test sets
def separate_dataset(start_path, train_path, test_path, caffe_file_path, train_perc, keyword_list=None,  species_class_id=None, field_name="Author"):
    create_folder(train_path)
    create_folder(test_path)
    field_species_files, unknowns_field_species_files = build_file_dictionary(start_path, field_name, keyword_list)

    #f_cvs, writer_cvs = open_new_cvs(os.path.join(caffe_file_path, "Dataset.cvs"))
    train_f = open_file(os.path.join(caffe_file_path, "Train.txt"))
    val_f = open_file(os.path.join(caffe_file_path, "Val.txt"))
    classes_f = open_file(os.path.join(caffe_file_path, "Classes.txt"))

    if species_class_id is None:
        species_class_id = {}
    print(species_class_id)

    for species_name in unknowns_field_species_files:
        if keyword_list is None or (not keyword_list is None and species_name in keyword_list ):
            idx, species_class_id = get_classes_idx(species_class_id, species_name)

            #write_list(classes_f, [idx, species_name])
            current_files = unknowns_field_species_files[species_name]
            total = len(current_files)
            train_number = int(total * train_perc)
            test_number = total - train_number
            shuffle_all_files(current_files)

            copy_files(train_path, current_files, idx, 0, train_number, train_f)
            copy_files(test_path, current_files, idx, train_number, total, val_f)


    for species_name in field_species_files:
        if keyword_list is None or (not keyword_list is None and species_name in keyword_list ):
            idx, species_class_id = get_classes_idx(species_class_id, species_name)

            total = len(field_species_files[species_name].keys())
            train_number = int(total * train_perc)
            test_number = total - train_number

            specimen_cont = 0
            for specimen_idx in field_species_files[species_name]:
                if specimen_cont < train_number:
                    copy_files(train_path, field_species_files[species_name][specimen_idx], idx, 0, len(field_species_files[species_name][specimen_idx]), train_f)
                else:
                    copy_files(test_path, field_species_files[species_name][specimen_idx], idx, 0, len(field_species_files[species_name][specimen_idx]), val_f)
                specimen_cont+=1

    write_class_dict(classes_f, species_class_id)
    close(classes_f)
    close(val_f)
    close(train_f)
    save_pickle(os.path.join(caffe_file_path, "ClassesIdx.pickle"), species_class_id)
    print("Dataset ready for Caffe")

'''
ORIGINAL_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/PlantCLEFStandard/Herbaria_CR_Matches_Good_Separation_Compressed/"
NEW_TRAIN_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/Caffe/Folders/Herbaria_CR_Matches_Good_Separation_Compressed/train/"
NEW_TEST_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/Caffe/Folders/Herbaria_CR_Matches_Good_Separation_Compressed/val/"
CAFFE_FILES_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/Caffe/Folders/Herbaria_CR_Matches_Good_Separation_Compressed/"
'''

'''
ORIGINAL_PATH = "/Users/maeotaku/Documents/Datasets/PlantCLEFStandard/ASUHerbariumCompressedGenusesK/"
NEW_TRAIN_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/ASUHerbariumCompressedGenusesK/train/"
NEW_TEST_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/ASUHerbariumCompressedGenusesK/val/"
CAFFE_FILES_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/ASUHerbariumCompressedGenusesK/"
'''
'''
ORIGINAL_PATH = "/opt/convnet_models/Datasets/Herbaria_Matches_PlantCLEF/"
NEW_TRAIN_PATH = "/opt/convnet_models/Caffe/Folders/Herbaria1K_PlantCLEF2015/train/"
NEW_TEST_PATH = "/opt/convnet_models/Caffe/Folders/Herbaria1K_PlantCLEF2015/dummy/"
CAFFE_FILES_PATH = "/opt/convnet_models/Caffe/Folders/Herbaria1K_PlantCLEF2015/"

classes = load_pickle("/opt/convnet_models/Caffe/Folders/PlantCLEF2015_train/ClassesIdx.pickle")
separate_dataset(ORIGINAL_PATH, NEW_TRAIN_PATH, NEW_TEST_PATH, CAFFE_FILES_PATH, 0.8, TaxaFilters.PLANTCLEF2015_COMMON_SPECIES_LIST, species_class_id=classes)
'''

ORIGINAL_PATH = "/data/carranza_expe/Datasets/Herbaria_PlantCLEF_Matches_Cropped"
NEW_TRAIN_PATH = "/data/carranza_expe/Caffe/Folders/Herbaria_PlantCLEF_Matches_Cropped/train/"
NEW_TEST_PATH = "/data/carranza_expe/Caffe/Folders/Herbaria_PlantCLEF_Matches_Cropped/test"
CAFFE_FILES_PATH = "/data/carranza_expe/Caffe/Folders/Herbaria_PlantCLEF_Matches_Cropped/"

separate_dataset(ORIGINAL_PATH, NEW_TRAIN_PATH, NEW_TEST_PATH, CAFFE_FILES_PATH, 0.7, TaxaFilters.PLANTCLEF2015_COMMON_SPECIES_LIST)

'''
ORIGINAL_PATH = "/opt/convnet_models/Datasets/PlantCLEF2015/test_separated"
NEW_TRAIN_PATH = "/opt/convnet_models/Caffe/Folders/PlantCLEF2015/val/"
NEW_TEST_PATH = "/opt/convnet_models/Caffe/Folders/PlantCLEF2015/dummy/"
CAFFE_FILES_PATH = "/opt/convnet_models/Caffe/Folders/PlantCLEF2015/"

classes = load_pickle("/opt/convnet_models/Caffe/Folders/PlantCLEF2015/ClassesIdx.pickle")
separate_dataset(ORIGINAL_PATH, NEW_TRAIN_PATH, NEW_TEST_PATH, CAFFE_FILES_PATH, 1, species_class_id=classes) #,TaxaFilters.CR_ASU_COMMON_SPECIES_LIST)
'''
