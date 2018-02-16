from DataSetsManagement import *
from TaxaFilters import TaxaFilters
from Files import *
from config import *


#PLANTCLEF file generation

'''
PLANTCLEF_TRAIN_PATH = r"/Datasets/RAW/PlantCLEF2017/data_cropped/"
PLANTCLEF_TRAIN_FILENAME = r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_train_cropped.hdf5"
PLANTCLEF_TRAIN_MEAN_FILENAME = r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_train_cropped.mean.pickle"
PLANTCLEF_HIERARCHIES_FILENAME = r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_hierarchies_cropped.pickle"

PLANTCLEF_TEST_PATH = r"/Datasets/RAW/PlantCLEF2017/test/"
PLANTCLEF_TEST_FILENAME = r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_test.hdf5"
PLANTCLEF_TEST_MEAN_FILENAME = r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_test.mean.pickle"
'''
'''
TRAIN_PATH = r"/Datasets/RAW/Herbaria255/Herbaria255_Images/"
TRAIN_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_train.hdf5"
TRAIN_MEAN_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_train.mean.pickle"
HIERARCHIES_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_hierarchies.pickle"

TEST_PATH = r"/Datasets/RAW/Herbaria255/test/"
TEST_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_test.hdf5"
TEST_MEAN_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_test.mean.pickle"
'''
'''
TRAIN_PATH = r"/Datasets/RAW/Herbaria_PlantCLEF_Matches_Typos/"
TRAIN_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_train.hdf5"
TRAIN_MEAN_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_train.mean.pickle"
HIERARCHIES_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_hierarchies.pickle"

TEST_PATH = r"/Datasets/RAW/Herbaria1K/test/"
TEST_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_test.hdf5"
TEST_MEAN_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_test.mean.pickle"
'''

TRAIN_PATH = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/RAW/"
TRAIN_FILENAME = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/HDF5/beers15_train.hdf5"
TRAIN_MEAN_FILENAME = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/HDF5/beers15_train.mean.pickle"
HIERARCHIES_FILENAME = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/HDF5/beers15_hierarchies.pickle"

TEST_PATH = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/RAW/"
TEST_FILENAME = r"/Users/maeotaku/Dropbox/Trabajo/Beer/AI/Data/HDF5/beers15_test.hdf5"
TEST_MEAN_FILENAME = r"/Users/maeotaku/Dropxbox/Trabajo/Beer/AI/Data/HDF5/beers15_hierarchies.pickle"

'''
PLANTCLEF_TRAIN_PATH = r"/Datasets/RAW/All_CR_Leaves_Cleaned/"
PLANTCLEF_TRAIN_FILENAME = r"/Datasets/HDF5/All_CR_Leaves_Cleaned/CR_Leaves_train.hdf5"
PLANTCLEF_TRAIN_MEAN_FILENAME = r"/Datasets/HDF5/All_CR_Leaves_Cleaned/CR_Leaves_train.mean.pickle"
PLANTCLEF_TRAIN_HIERARCHIES_FILENAME = r"/Datasets/HDF5/All_CR_Leaves_Cleaned/CR_Leaves_hierarchies.pickle"
'''

train = PlantCLEFHDF5(TRAIN_PATH, RESIZE_SIZE, TRAIN_FILENAME, ["Class"], hierarchy_params_file=HIERARCHIES_FILENAME, root_has_subfolders=True, include_labels=True)
train.open_HDF5(100)
train.generate_HDF5(100)
train.close_HDF5()
train.save_hierarchies()
hierarchy_params = train.hierarchy_params
print(hierarchy_params)
'''
PLANTCLEF_train = PlantCLEFHDF5(PLANTCLEF_TRAIN_PATH, RESIZE_SIZE, PLANTCLEF_TRAIN_FILENAME, level_list=["ClassId", "Genus", "Family"], root_has_subfolders=True, include_labels=True, hierarchy_params=hierarchy_params)
PLANTCLEF_train.open_HDF5(100)
PLANTCLEF_train.generate_HDF5(100)
PLANTCLEF_train.close_HDF5()
#PLANTCLEF_train.reassign_hierarchies()
'''





f_train, X_train, y_train = open_hdf5_file(TRAIN_FILENAME)
#f_test, X_test, y_test = open_hdf5_file(TEST_FILENAME)
#show_data_shapes(X_train, y_train, X_test, y_test, X_test, y_test)

shape = (CHANNELS, RESIZE_SIZE, RESIZE_SIZE)
#test_mean = mean(X_test, shape)
#test_stdev = stdev(X_test, test_mean, shape)
train_mean = mean(X_train, shape)
train_stdev = stdev(X_train, train_mean, shape)

save_pickle(TRAIN_MEAN_FILENAME, { "mean" : train_mean, "stdev" : train_stdev} )
#save_pickle(TEST_MEAN_FILENAME, { "mean" : test_mean, "stdev" : test_stdev} )

'''
f = h5py.File(PLANTCLEF_TRAIN_FILEOUTPUT, 'r')
X = f["images"]
y = f["labels"]
#y = y[:, 1].reshape(y.shape[0])
print(X.shape, y.shape)

for inputs, targets in mini_batch_iterate(X, y, batch_size=64, img_size=256, normalize=True):
    for img in inputs:
        #img = inputs[3]
        print(img)
        #rimg = DLDataAug.rotate_and_crop(img, 20)
        show_img(img)
    #show_img(rimg)
'''
