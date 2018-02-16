from common import *

def parse_class_file(class_file_path):
    classes = load_pickle(class_file_path)
    return classes

def parse_file_pointer_file(file_pointer_file_path):
    f = open_file(file_pointer_file_path, "r")
    files = {}
    for line in f:
        words = line.split(" ")
        filename = ' '.join(words[:len(words)-1])
        classnumber = int(words[len(words)-1])
        if not classnumber in files:
            files[classnumber] = [filename]
        else:
            files[classnumber] += [filename]
    return files

def invert_dict(dict):
    n_dict = {}
    for key in dict.keys():
        n_key = dict[key]
        n_dict[n_key] = key
    return n_dict

def copy_subset(subset_files, input_path, output_path, inverted_dict):
    for classnumber in subset_files.keys():
        species_name = inverted_dict[classnumber]
        path = os.path.join(output_path, species_name)
        create_folder(path)
        complete_path_files = []
        for filename in subset_files[classnumber]:
            complete_path_files += [ os.path.join(input_path, filename) ]
        copy_files(path, complete_path_files, classnumber, 0, len(subset_files[classnumber]))


def restore_dataset(caffe_files_path, train_path, test_path):
    classes = parse_class_file(os.path.join(caffe_files_path, "ClassesIdx.pickle"))
    trainings = parse_file_pointer_file(os.path.join(caffe_files_path, "Train.txt"))
    testings = parse_file_pointer_file(os.path.join(caffe_files_path, "Val.txt"))
    inv_classes = invert_dict(classes)
    print(trainings)

    create_folder(train_path)
    create_folder(test_path)

    copy_subset(trainings, os.path.join(caffe_files_path, "train"), train_path, inv_classes)
    copy_subset(testings, os.path.join(caffe_files_path, "test"), test_path, inv_classes)

    print("Dataset ready and separated")


CAFFE_FILES_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/CR_Leaves_Unbiased/"
NEW_TRAIN_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/CR_Leaves_Unbiased_TRAIN/"
NEW_TEST_PATH = "/Users/maeotaku/Documents/Datasets/Caffe/Folders/CR_Leaves_Unbiased_TEST/"

restore_dataset(CAFFE_FILES_PATH, NEW_TRAIN_PATH, NEW_TEST_PATH)
#delete_files_CRNationalMuseum("/Users/maeotaku/OneDrive/PhD_Research/Datasets/All_CR_Leaves_Segmented2/")
