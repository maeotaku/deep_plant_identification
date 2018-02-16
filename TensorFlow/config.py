#sudo THEANO_FLAGS="device=gpu" python run_model_multi_label.py

'''
>>> h = open(r"/Datasets/HDF5/PlantCLEF2017/PlantCLEF2017_hierarchies.pickle", "rb")
>>> f = pickle.load(h)
>>> f['levels'].keys()
['ClassId', 'Genus', 'Family']
>>> len(f['levels']['Genus'])
2991
>>> len(f['levels']['Family'])
341
>>> len(f['levels']['ClassId'])
10000
'''

'''
>>> h = open(r"/Datasets/HDF5/Herbaria255/Herbaria255_hierarchies.pickle", "rb")
>>> import pickle
>>> f = pickle.load(h)
>>> f['levels'].keys()
['Genus', 'Species', 'Family']
>>> len(f['levels']['Genus'])
158
>>> len(f['levels']['Species'])
203
'''

'''
>>> h = open(r"/Datasets/HDF5/Herbaria1K/Herbaria1K_hierarchies.pickle", "rb")
>>> f = pickle.load(h)
>>> import pickle
>>> f = pickle.load(h)
>>> len(f['levels']['Genus'])
471 498
>>> len(f['levels']['Family'])
134 124
>>> len(f['levels']['Species'])
1204  1191
'''


#theano
#dev version is the best:
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#tmux
#tmux new -s training
#exit: ctlr+b d
#tmux attach -t training

#NVIDIA watcher
#watch -n 1 nvidia-smi

#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-8.0/bin:$PATH


#docker
#for id: docker ps -a
#to run it: nvidia-docker run -it -v /home/carranza/code/:/home/carranza/code/ -v /opt/:/opt/ kaixhin/cuda-theano
#to commit changes: docker commit 96ba30bffc51 kaixhin/cuda-theano

#pastalog:
#pip install pastalog
#pastalog --install
#pastalog --serve 8120

#connect ot server
#ssh carranza@tilleul.cirad.fr

#rsync
#rsync -a jcarranza@138.91.165.45:/home/jcarranza/Downloads/PlantCLEFCrawler/Config/ASUHerbarium/Output Output/

#libgpuarray
#cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda-7.5/include -DCUDA_CUDA_LIBRARY=/usr/local/cuda-7.5/lib64/stubs/libcuda.so
#test on 2 GPUs: THEANO_FLAGS="contexts=dev0->cuda0;dev1->cuda1,device=cuda" python run_model.py

#sudo rsync -a -e "ssh -p 443" goeau@otmedia.lirmm.fr:/data/plantnet/LifeClef/LifeCLEF/GoingDeeperHerbarium/Herbaria255_Images.zip /Datasets/RAW/Herbaria255


import time

def get_current_time_file_name():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr
time_file_name = get_current_time_file_name()
LOG_FILE = "log_file_Herbaria1K_multi-label_{}.log".format(time_file_name)
CLASS_LOG_FILE = "log_file_Herbaria1K_classes_multi-label_{}.log".format(time_file_name)
#LOG_FILE = "log_file_PlantCLEF17_baseline_organs_{}.log".format(get_current_time_file_name())
PASTA_LOG_SERVER = "195.221.175.173:8120"
RUN_MODE = 'gpu'
#TEST_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/PlantCLEF2015/PlantCLEF2015TestDataWithAnnotations/"
#TRAIN_PATH = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/PlantCLEF2015/train/"

'''
#plantclef2017
TRAIN_FILENAME = "PlantCLEF2017_train.hdf5"
TEST_FILENAME = "PlantCLEF2017_test.hdf5"
VALIDATION_FILENAME = ""
HIERARCHIES_FILENAME = "PlantCLEF2017_hierarchies.pickle"
PARAMS_FILENAME = "PlantCLEF2017.npz"
TRAIN_MEAN_FILENAME = "PlantCLEF2017_train.mean.pickle"
TEST_MEAN_FILENAME = "PlantCLEF2017_test.mean.pickle"

#plantclef2015
TRAIN_FILENAME = "DataSets/PlantCLEF2015_train.hdf5"
TEST_FILENAME = "DataSets/PlantCLEF2015_test.hdf5"
VALIDATION_FILENAME = ""
HIERARCHIES_FILENAME = "DataSets/PlantCLEF2015_hierarchies.pickle"
PARAMS_FILENAME = "DataSets/PlantCLEF2015.npz"
TRAIN_MEAN_FILENAME = "DataSets/PlantCLEF2015_train.mean.pickle"
TEST_MEAN_FILENAME = "DataSets/PlantCLEF2015_test.mean.pickle"
'''

'''
TRAIN_FILENAME = "/Datasets/HDF5/Herbaria255/Herbaria255_train.hdf5"
TEST_FILENAME = ""
HIERARCHIES_FILENAME = r"/Datasets/HDF5/Herbaria255/Herbaria255_hierarchies.pickle"
#PARAMS_FILENAME = "PlantCLEF2017_baseline_weights_{}.npz"
PARAMS_FILENAME = "Herbaria255_hierarchical_weights_{}.npz"
TRAIN_MEAN_FILENAME = "/Datasets/HDF5/Herbaria255/Herbaria255_train.mean.pickle"
TEST_MEAN_FILENAME = ""
#CLASSES_FILENAME="/Code/Theano/DataSets/CRMuseumLeaves_hierarchies_species.pickle"
'''

TRAIN_FILENAME = "/Datasets/HDF5/Herbaria1K/Herbaria1K_train.hdf5"
TEST_FILENAME = ""
HIERARCHIES_FILENAME = r"/Datasets/HDF5/Herbaria1K/Herbaria1K_hierarchies.pickle"
#PARAMS_FILENAME = "PlantCLEF2017_baseline_weights_{}.npz"
PARAMS_FILENAME = "Herbaria1K_hierarchical_weights_{}.npz"
TRAIN_MEAN_FILENAME = "/Datasets/HDF5/Herbaria1K/Herbaria1K_train.mean.pickle"
TEST_MEAN_FILENAME = ""


#INIT_WEIGHTS = "TrainedWeights/inception_v3.pkl"
INIT_WEIGHTS = 'TrainedWeights/blvc_googlenet.pkl'
#INIT_WEIGHTS = "CRLeaves/CRLeaves_20170330-023958.npz"
#INIT_WEIGHTS = 'TrainedWeights/resnet50.pkl'
#'TrainedWeights/vgg_cnn_s.pkl'
IMG_SIZE = 256
RESIZE_SIZE = 224 #-1 if no resize want to be applied to images at all
CHANNELS = 3


BATCH_SIZE = 32 #32 images for each batch
MAX_QUEUE_BATCHES = 100#120#100 #maximum number of batches in the queue
TRAIN_ITERATION_SIZE = 1600#120#3200 #how many batches for 1 training iteration = train size / batch size
VAL_EACH_X_ITERATIONS = 100#120#250 #to validate based on this amount of iterations not on epoch
VAL_ITERATION_SIZE = 1000#20#1000 #how many batches for 1 iteration validation = val size / batch size
TEST_ITERATION_SIZE = 1000#20#1000 #how many batches for 1 iteration testing = test size / batch size
EPOCHS = 5#20#5
'''
BATCH_SIZE = 32 #32 images for each batch
MAX_QUEUE_BATCHES = 100 #maximum number of batches in the queue
TRAIN_ITERATION_SIZE = 6400 #how many batches for 1 training iteration = train size / batch size
VAL_ITERATION_SIZE = 1600 #how many batches for 1 iteration validation = val size / batch size
TEST_ITERATION_SIZE = 30 #how many batches for 1 iteration testing = test size / batch size
EPOCHS = 10
'''

TOP_GUESSES=10

BATCH_NORMALIZE = True
WEIGHT_DECAY = 0.0002
MOMENTUM = 0.9
BASE_LEARNING_RATE = 0.0075

TOTAL_CLASSES = 1000
NEW_TOTAL_CLASSES = 1191#203#1191 #341
#NEW_TOTAL_CLASSES = 255#para correr el modelo de costa rica
