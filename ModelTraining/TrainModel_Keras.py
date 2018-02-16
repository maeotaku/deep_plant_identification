# Simple Convolutional Neural Network classifier for dogs and cats images


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

NUM_CLASSES=3
EPOCH=5
batch_size = 10

def demo_normalize(path):
	img = Image.open(path).convert('RGB')
	img=img.resize((64,64))
	arr = np.array(img)
	return arr

def add_new_last_layer(base_model, nb_classes):
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
	model = Model(input=base_model.input, output=predictions)
	return model

def setup_to_transfer_learn(model, base_model):
	for layer in base_model.layers:
		layer.trainable = False
	model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/jvillacis/data/simple_leaves/train',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory('/home/jvillacis/data/simple_leaves/test',
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
#Print class indices
print('Class indices: ', training_set.class_indices);



base_model = InceptionV3(weights='imagenet', include_top=False)
#model.summary(line_length=150)
model = add_new_last_layer(base_model, NUM_CLASSES)
setup_to_transfer_learn(model, base_model)





model.fit_generator(training_set,
                         steps_per_epoch = 80000 // batch_size,
                         epochs = EPOCH,
                         validation_data = validation_set,
                         validation_steps = 20000 // batch_size)

#Save trained weights
model.save_weights('/home/jvillacis/leaves/simple_32/inception_weights.h5', overwrite = True)
"""
NUM_CLASSES=3
EPOCH=10
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = NUM_CLASSES, activation = 'softmax'))

classifier.load_weights("leaves_prueba.h5")
classifier.layers.pop()
classifier.layers.pop()
classifier.layers.pop()
classifier.layers.pop()
#classifier.layers.pop()
#classifier.layers.pop()
classifier.summary(line_length=150)

dec=classifier.layers[-1].output
dec2=Conv2DTranspose(32,(3,3),padding="same",activation="relu",name='deconv')(dec)

model2=Model(input=classifier.input,output=dec2)
model2.summary(line_length=150)

model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

#Define batch size
batch_size = 10


cat=demo_normalize('/home/jvillacis/data/simple/test/annona/annona.jpg')
cat_batch=np.expand_dims(cat,axis=0)

conv1_layer_model = Model(inputs=model2.input,

                                outputs=model2.get_layer('deconv').output)



conv1 = conv1_layer_model.predict(cat_batch)

conv1=np.squeeze(conv1,axis=0)

conv1 = np.rollaxis(conv1, -1)

conv1=conv1[0]

#fig = plt.figure()
print(conv1)
plt.imshow(conv1)

plt.savefig('conv1.png')

"""
