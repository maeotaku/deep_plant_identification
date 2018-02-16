#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12
DATASET_TYPE=val
#where database is stored
EXAMPLE=/opt/convnet_models/Caffe/LMDB/Herbaria_Matches_PlantCLEF/$DATASET_TYPE/
#where train and val text files are located
DATA=/opt/convnet_models/Caffe/Folders/Herbaria_Matches_PlantCLEF/
#where the compiled caffe tools are available
TOOLS=/home/goeau/caffe/build/tools/

$TOOLS/compute_image_mean $EXAMPLE/ \
  $DATA/$DATASET_TYPE.mean.binaryproto

echo "Done."
