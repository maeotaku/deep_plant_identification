from common import *

def delete_files_CRNationalMuseum(start_path):
    delete_files_with_pattern_name(start_path, r'.*[1-9]_[1-9]_3.*')

def get_specimen_number_CRNationalMuseum(filename):
    matchObj = re.match(r'.*([1-9])_[1-9]_[1-9].*', filename)
    if matchObj:
        return matchObj.group(1)
    return None

def build_file_dictionary(start_path):
    specimen_files = {}
    unknowns_specimen_files = {}
    for species_name in get_immediate_subdirectories(start_path):
        specimen_files[species_name] = {}
        unknowns_specimen_files[species_name] = []
        species_folder = os.path.join(start_path, species_name)
        for filename in get_file_list(species_folder):
            _, extension = os.path.splitext(filename)
            if extension.upper() in [".JPG", ".JPEG", ".BMP", ".PNG"]:
                specimen_number = get_specimen_number_CRNationalMuseum(filename)
                complete_path = os.path.join(species_folder, filename)

                unknowns_specimen_files[species_name].append(complete_path)

                if specimen_number == 'X':
                    unknowns_specimen_files[species_name].append(complete_path)
                else:
                    if specimen_number in specimen_files[species_name]:
                        specimen_files[species_name][specimen_number].append(complete_path)
                    else:
                        specimen_files[species_name][specimen_number] = [ complete_path ]

            else:
                print("Ignoring file ", filename)
    return specimen_files, unknowns_specimen_files

#separate in train and test sets
def separate_dataset(start_path, train_path, test_path, caffe_file_path, train_perc, keyword_list=None, species_class_id=None):
    create_folder(train_path)
    create_folder(test_path)
    specimen_files, unknowns_specimen_files = build_file_dictionary(start_path)

    train_f = open_file(os.path.join(caffe_file_path, "Train.txt"))
    val_f = open_file(os.path.join(caffe_file_path, "Val.txt"))
    classes_f = open_file(os.path.join(caffe_file_path, "Classes.txt"))

    if species_class_id is None:
        species_class_id = {}

    for species_name in unknowns_specimen_files:
        idx, species_class_id = get_classes_idx(species_class_id, species_name)

        #write_list(classes_f, [idx, species_name])
        current_files = unknowns_specimen_files[species_name]
        total = len(current_files)
        train_number = int(total * train_perc)
        test_number = total - train_number
        shuffle_all_files(current_files)

        copy_files(train_path, current_files, idx, 0, train_number, train_f)
        copy_files(test_path, current_files, idx, train_number, total, val_f)


    for species_name in specimen_files:
        idx, species_class_id = get_classes_idx(species_class_id, species_name)

        total = len(specimen_files[species_name].keys())
        train_number = int(total * train_perc)
        test_number = total - train_number

        specimen_cont = 0
        for specimen_idx in specimen_files[species_name]:
            if specimen_cont < train_number:
                copy_files(train_path, specimen_files[species_name][specimen_idx], idx, 0, len(specimen_files[species_name][specimen_idx]), train_f)
            else:
                copy_files(test_path, specimen_files[species_name][specimen_idx], idx, 0, len(specimen_files[species_name][specimen_idx]), val_f)
            specimen_cont+=1

    write_class_dict(classes_f, species_class_id)
    close(classes_f)
    close(val_f)
    close(train_f)
    save_pickle(os.path.join(caffe_file_path, "ClassesIdx.pickle"), species_class_id)
    print("Dataset ready for Caffe")


ORIGINAL_PATH = "/Datasets/RAW/All_CR_Leaves_Cleaned/"
NEW_TRAIN_PATH = "/Datasets/Separated/Folders/CR_Leaves_Biased/train/"
NEW_TEST_PATH = "/Datasets/Separated/Folders/CR_Leaves_Biased/test/"
CAFFE_FILES_PATH = "/Datasets/Separated/Folders/CR_Leaves_Biased/"


separate_dataset(ORIGINAL_PATH, NEW_TRAIN_PATH, NEW_TEST_PATH, CAFFE_FILES_PATH, 0.7) #,TaxaFilters.CR_ASU_COMMON_SPECIES_LIST)
#delete_files_CRNationalMuseum("/Users/maeotaku/OneDrive/PhD_Research/Datasets/All_CR_Leaves_Segmented2/")
