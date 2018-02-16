from __future__ import print_function

import sys
import os
import io
import time
import random
random.seed(time.time())

from collections import deque

import numpy as np
from random import shuffle

from scipy import misc
from skimage.io import imread

import h5py
from lxml import etree

import glob

import pickle


from Files import *
from HierarchyManagement import *
from Visualizations import *
from DataAugmentation import *
from config import *

def show_data_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
	print("Train data: ", X_train.shape)
	print("Train labels: ", y_train.shape)
	print("Validation data: ", X_val.shape)
	print("Validation labels: ", y_val.shape)
	print("Test data: ", X_test.shape)
	print("Test labels: ", y_test.shape)

def numpy_to_string(img):
	try:
		return np.ndarray.tostring(img)
	except Exception as e:
		print('Cannot convert image to string ', e)
		return None

def string_to_numpy(byte_string, shape):
	try:
		return np.fromstring(byte_string, np.uint8).reshape(shape)
	except Exception as e:
		print("Cannot convert string to numpy ", e)
		return None

def create_empty_data_labels(batch_size, channels, img_size, classes_level_list_size):
	images = np.zeros((batch_size, channels, img_size, img_size), dtype='float32')
	targets = np.zeros((batch_size, classes_level_list_size), dtype='uint32')
	return images, targets

def randomize_batch(batch, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_batch = batch[permutation, :, :, :]
	if labels.ndim == 1:
		shuffled_labels = labels[permutation]
	else:
		shuffled_labels = labels[permutation, :]
	return shuffled_batch, shuffled_labels

def stdev(dataset, mean, shape):
	#assert(len(shape) == 3)
	stdev = np.zeros(shape, dtype=np.float32)
	n = len(dataset)
	cont=0
	while cont < n:
		img = dataset[cont]
		stdev += ((img - mean) ** 2)
		if cont % 10000 == 0:
			print("Calculating stdev ", cont, n)
		cont+=1
	stdev = np.sqrt(stdev / n)
	stdev[stdev == 0] = 0.00000001
	return stdev


def mean(dataset, shape):
	#assert(len(shape) == 3)
	m = np.zeros(shape, dtype=np.float32)
	n = len(dataset)
	cont=0
	while cont < n:
		img = dataset[cont]
		m += img
		if cont % 10000 == 0:
			print("Calculating mean ", cont, n)
		cont+=1
	m = m / n
	return m

def normalize_batch(batch, mean, stdev):
	for i in range(batch.shape[0]):
		batch[i] = (batch[i] - mean) / (stdev + 0.000001)
	return batch
	#np.float32(batch) / np.float32(255)

def reorder_img(img):
	try:
		img = np.swapaxes(img, 0, 1)
		img = np.swapaxes(img, 0, 2)
		return img
	except Exception as e:
		print("Cannot resize image", e)
		return None

def resize_mean_stdev(mean, stdev, size, new_size, channels):
	if not mean is None and not stdev is None:
		if size!=new_size:
			print(mean.shape); mean = misc.imresize(mean, (new_size, new_size, channels), 'bilinear', mode=None)
			stdev = misc.imresize(stdev, (new_size, new_size, channels), 'bilinear', mode=None)
	return reorder_img(mean), reorder_img(stdev)
	#np.float32(batch) / np.float32(255)



def randomly_generate_augmentation(batch_images, batch_targets, batch_size, img_size, max_number_of_images, shuffle=True):
	augments = [ AugmentationTypes.MirrorVertical, AugmentationTypes.MirrorBoth ]
	images, targets = create_empty_data_labels(batch_images.shape[0] * max_number_of_images, 3, img_size, batch_targets.shape[1])
	while current_item < batch_size:
		cont = 0
		while cont < max_number_of_images:
			aug_type = random.choice(augments)
			cont+=1
		current_item+=1
	if shuffle:
		return randomize_batch(images, targets)
	else:
		return images, targets

def crop_batch_if_needed(batch_images, img_size):
	original_size = batch_images.shape[3]
	if original_size > img_size:
		return batch_images[:, :, 0:img_size, 0:img_size]
	return batch_images

def check_num_dims(some_array, ndim):
	if not isinstance(some_array, np.ndarray) and ndim==1:
		return True
	if some_array.ndim == ndim:
		return True
	return False

def generate_new_indexes(dataset_images, dataset_labels):
	return np.array(range(0, len(dataset_images)))

def get_random_indexes(indexes, size):
	if size <= len(indexes):
		random_indexes = random.sample(range(0, len(indexes)), size)
		sample_indexes = indexes[random_indexes]
		indexes = np.delete(indexes, random_indexes, 0)
		return indexes, sample_indexes
	else: #last left
		return np.array([]), indexes

'''
Separates a dataset in subsets randomly, given percentages, index-wise. Returns a dictionary with the indexes of the subsets
X is the data, y are the targets
data_distribution is a dictionary with the format { name : percentage } like { "train" : 0.8, "test" : 0.2}
'''
def generate_separated_indexes(X, y, data_distribution):
	assert X.shape[0] == y.shape[0]
	total = y.shape[0]
	print("Total items in X ", total)
	indexes = generate_new_indexes(X, y)
	separated_indexes = {}
	total_subsets = len(data_distribution.keys())
	cont=0
	while cont < total_subsets:
		subset_desc_key = data_distribution.keys()[cont]
		perc = data_distribution[subset_desc_key]
		if cont == total_subsets - 1: #last subset_desc_key
			separated_indexes[subset_desc_key] = indexes
		else:
			size = int(total * perc)
			print("Extracting subset of size ", size)
			indexes, samples_indexes = get_random_indexes(indexes, size)
			separated_indexes[subset_desc_key] = samples_indexes
		cont+=1
	return separated_indexes

#Gets a new batch completely random from the original dataset, deleting those indexes from the index list
def get_new_random_batch(dataset_images, dataset_labels, indexes, batch_size):
	indexes, batch_indexes = get_random_indexes(indexes, batch_size)
	#print(indexes)
	batch_images, batch_targets = create_empty_data_labels(batch_indexes.shape[0], dataset_images.shape[1], dataset_images.shape[2], dataset_labels.shape[1])
	cont=0
	for index in batch_indexes:
		batch_images[cont, :, :, :] = dataset_images[index , :, :, :]
		batch_targets[cont, :] = dataset_labels[index, :]
		cont+=1
	return batch_images, batch_targets, indexes

#fills a que with all the batches for the maximum number of iterations
def fill_batches_queue(batch_queue, dataset_images, dataset_labels, batch_size, max_iter, channels=3, img_size=256, classes_level_list_size=1, normalize=True, original_indexes=None, mean=None, stdev=None):
	assert(len(dataset_images) == len(dataset_labels))

	def stop_batching(indexes, current_batch, max_iter):
		if max_iter == -1:
			if indexes.shape[0] <= 0:
				return True
			else:
				return False
		else:
			des = current_batch > max_iter
			if des:
				print("Queue reached the fulliest...", current_batch - 1, indexes.shape)
			#if not indexes.shape[0] <= 0:
			return des

	if original_indexes is None:
		indexes = generate_new_indexes(dataset_images, dataset_labels)
	else:
		indexes = original_indexes[:] #make a copy we dont want to lose the indexes
	#print(indexes.shape)
	batch_count = 0
	while not stop_batching(indexes, batch_count, max_iter):
		#if batch_count % 2 == 0 and batch_count > 0:
		if len(batch_queue) <= MAX_QUEUE_BATCHES:
			#sstart_time = time.time()
			images, targets, indexes = get_new_random_batch(dataset_images, dataset_labels, indexes, batch_size)
			#print(indexes.shape)
			if not indexes.shape[0] <= 0:
				#print(images.shape, indexes.shape)
				images = crop_batch_if_needed(images, img_size)
				if not mean is None and not stdev is None:
					images = normalize_batch(images, mean, stdev)
				images, targets = randomize_batch(images, targets) #last shuffle
				batch_queue.append( ( images.astype(np.float32), targets.astype(np.int32) ) )
				#print("Training batch took {:.3f}s".format(time.time() - sstart_time))
				#print("Added {} batches from {} to the queue. Currently in queue {}".format(batch_count, max_iter, len(batch_queue)))
			batch_count+=1



#iterate randomly over the dataset, yielding batches of data and labels
def mini_batch_iterate(dataset_images, dataset_labels, batch_size, channels=3, img_size=256, classes_level_list_size=1, normalize=True, test_interval=-1, original_indexes=None):

	def stop_batching(indexes, current_batch, test_interval):
		if test_interval == -1:
			if indexes.shape[0] <= 0:
				return True
			else:
				return False
		else:
			return current_batch >= test_interval or indexes.shape[0] <= 0

	assert(len(dataset_images) == len(dataset_labels))
	if original_indexes is None:
		indexes = generate_new_indexes(dataset_images, dataset_labels)
	else:
		indexes = original_indexes[:] #make a copy we dont want to lose the indexes
	batch_count = 0
	while not stop_batching(indexes, batch_count, test_interval):
		#sstart_time = time.time()
		images, targets, indexes = get_new_random_batch(dataset_images, dataset_labels, indexes, batch_size)
		images = crop_batch_if_needed(images, img_size)
		if normalize:
			images = normalize_batch(images)
		images, targets = randomize_batch(images, targets) #last shuffle
		#print(images[5])
		yield images.astype(np.float32), targets.astype(np.uint8)
		#print("Training batch took {:.3f}s".format(time.time() - sstart_time))
		batch_count+=1

#iterate randomly over the dataset, yielding batches of data and labels from the queue
def queued_mini_batch_iterate(batch_queue, batch_quantity):
	batch_count = 0
	while batch_count < batch_quantity:
		if(len(batch_queue) > 0):
			images, targets = batch_queue.popleft()
			yield images, targets
			batch_count+=1
	#else:
		#print(batch_count, test_interval)


def all_unique(x):
	seen = set()
	return not any(i in seen or seen.add(i) for i in x)


class PlantCLEFHDF5():
	BATCH_LABEL = 'batch'
	CHANNEL_LABEL = 'channel'
	HEIGHT_LABEL = 'height'
	WIDTH_LABEL = 'width'

	HIERARCHY_LABEL = "Labels"



	IMAGE_EXTENSIONS = [".JPG", ".PNG", ".jpg", ".png"]
	CHANNELS = 3



	'''
	start_path refers to the path where the images are stored with their xmls with the PlantCLEF standard
	#file_output is a dictionary like {name : percentage}, useful to subdivide the data set into subsets, such as { "train" : 50, "test" : 50} the HDF5 file
	file_output is the name of the hdf5 file
	img_size is the size of the final images once converted
	level_list is a list of labels for each level of the classes_level_list, or even non related labels such as genus, species, family, etc.
	root_has_subfolders means the root folder has subfolders so needs to be threated a bit differently, most of the times each subfolder is a species
	include_labels enabled sabing the labels or targets y. So if it is false, X is saved in the file and not y.
	classes_level_list is a dictionary containing the different hiearchy levels for the labels, if one wants to use a previous classes_level_list from another dataset, i.e share species ids between datasets
	'''
	def __init__(self, start_path, img_size, file_output, level_list, hierarchy_params_file=None, root_has_subfolders=False, include_labels=True, keyword_list=None, hierarchy_params=None):
		self.include_labels = include_labels
		self.root_has_subfolders = root_has_subfolders
		self.hierarchy_params_file = hierarchy_params_file
		self.level_list = level_list
		self.level_size = len(level_list)
		#hierarchies
		self.initialize_hierarchies(hierarchy_params)
		self.item_number = self.get_number_of_files(start_path) / 2 #since we have the PlantCLEF stanrd with 1 xml per image
		self.img_size = img_size
		self.start_path = start_path
		self.file_output = file_output
		self.file_list = []
		self.item_number = 0
		self.build_file_dictionary(start_path)
		print(self.hierarchy)
		self.filter_file_list_by_keywords(keyword_list) #filter certain files only, based on taxa maybe
		self.item_number = len(self.file_list)
		self.shuffle_all_files()
		print("Dataset size: ", self.item_number)

	def initialize_hierarchies(self, hierarchy_params):
		self.hierarchy_params, self.classes_level_list, self.hierarchy, self.inv_hierarchy = initialize_hierarchies(self.hierarchy_params_file, self.level_list, hierarchy_params)

	def reassign_hierarchies(self):
		self.hierarchy_params = {}
		self.hierarchy_params[HIERARCHY_FILE_PARAM_LEVELS] = self.classes_level_list
		self.hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY] = self.hierarchy
		self.hierarchy_params[HIERARCHY_FILE_PARAM_HIERARCHY_INV] = self.inv_hierarchy

	def save_hierarchies(self):
		self.reassign_hierarchies()
		save_hierarchies(self.hierarchy_params_file, self.hierarchy_params)

	def add_class_on_level(self, class_level, class_name):
		return add_class_on_level(self.classes_level_list, class_level, class_name)

	def add_to_hierarchy(self, attrs):
		add_to_hierarchy(self.hierarchy, self.inv_hierarchy, attrs)

	def shuffle_all_files(self):
		shuffle(self.file_list)

	def read_image_on_numpy(self, file_path):
		try:
			img = misc.imread(file_path, mode='RGB')
			return img
		except Exception as e:
			print("Cannot load image", e)
			return None

	def resize_img_size_size_channels(self, img):
		try:
			#print(img.shape, self.img_size, (self.img_size, self.img_size, self.CHANNELS))
			resized = misc.imresize(img, (self.img_size, self.img_size, self.CHANNELS), 'bilinear', mode=None)
			#print(resized.shape)
			return resized
		except Exception as e:
			print("Cannot resize image", e)
			return None



	def build_file_attrs(self, file_path):
		try:
			parser = etree.XMLParser(encoding='utf-8')
			xmlText = open(file_path, "r").read().replace("&", "&amp;")
			doc = etree.parse( io.BytesIO(xmlText), parser )
			#doc = etree.parse(file_path)
			attrs = []
			for level_label in self.level_list:
				elem = doc.find(level_label)
				idx = self.add_class_on_level(level_label, elem.text)
				if elem is None or elem.text is None or idx is None:
					return None
				attrs.append(idx)
			self.add_to_hierarchy(attrs)
			return np.array(attrs, np.uint32)
		except Exception as e:
			print("Cannot read xml file", file_path, e)
			return None

	def get_number_of_files(self, path):
		file_count = 0
		for file in glob.glob(os.path.join(path, '*.*')):
			file_count += 1
		return file_count

	#useful to filter a list of file names by genus or species names
	def filter_file_list_by_keywords(self, keyword_list=None):
		if keyword_list is None:
			return self.file_list
		print("Filtering file list. Original size", len(self.file_list))
		filtered_file_list = []
		for filename in self.file_list:
			upper_filename = filename.upper()
			for keyword in keyword_list:
				if keyword.upper() in upper_filename:
					filtered_file_list.append(filename)
					break;
		print("Filtered file list size", len(filtered_file_list))
		self.file_list = filtered_file_list



	def build_file_dictionary(self, start_path):
		print("Catching file list...")
		if not self.root_has_subfolders:
			print("No subfolers...", start_path)
			for path, _, files in os.walk(start_path):
				for filename in files:
					name, extension = os.path.splitext(filename)
					if extension.upper() in self.IMAGE_EXTENSIONS:
						print("Adding ", filename)
						self.file_list.append(os.path.join(path, filename))
		else:
			print("Checking subfolders of", start_path)
			for species_name in get_immediate_subdirectories(start_path):
				print("Subfolder", species_name)
				species_folder = os.path.join(start_path, species_name)
				for filename in get_file_list(species_folder):
					_, extension = os.path.splitext(filename)
					if extension.upper() in self.IMAGE_EXTENSIONS:
						print("Adding ", filename)
						self.file_list.append(os.path.join(species_folder, filename))

	def generate_HDF5(self, batch_size):
		print("Generating HDF5 file...")
		t0 = time.time()
		item_count = 1
		overall_item_count = 0
		images, targets = create_empty_data_labels(batch_size, self.CHANNELS, self.img_size, self.level_size)
		self.real_file_quantity = 0
		for complete_img_filename in self.file_list:
			#print("Processing file... ", complete_img_filename)
			process = True
			img = self.read_image_on_numpy(complete_img_filename)
			if not img is None:
				if self.include_labels:
					name, _ = os.path.splitext(complete_img_filename)
					complete_label_filename = os.path.join(self.start_path, name + ".xml")
					attrs = self.build_file_attrs(complete_label_filename)
					if attrs is None:
						print("Empty attributes found in ", complete_label_filename)
						process = False
					else:
						targets[item_count-1, :] = attrs
						#print(targets[item_count-1, :])
				if process:
					img = self.resize_img_size_size_channels(img)
					img = reorder_img(img)
					images[item_count-1,:,:,:] = img
					#print(images[item_count-1,:,:,:], images[item_count-1,:,:,:].shape)
					if (item_count % batch_size == 0):
						t1 = time.time()
						self.images[overall_item_count : overall_item_count + batch_size , :, :, :] = images
						if self.include_labels:
							self.targets[overall_item_count : overall_item_count + batch_size, :] = targets
						images, targets = create_empty_data_labels(batch_size, self.CHANNELS, self.img_size, self.level_size)
						overall_item_count+=batch_size
						print(overall_item_count, "/", self.item_number, "Batch took", t1 - t0, " seconds.")
						item_count=0
						t0 = time.time()
					#else:
					item_count+=1
		#print("Lenght of levels: ", all_unique(self.classes_level_list["ClassId"]), all_unique(self.classes_level_list["Genus"]), all_unique(self.classes_level_list["Family"]) )
		self.item_number = item_count - 1
		print("HDF5 file created.")


	def open_HDF5(self, batch_size):
		print("Opening HDF5 file...")
		self.f = h5py.File(self.file_output, mode='w', compression="gzip", compression_opts=9)
		#self.f = h5py.File(self.file_output, mode='w', compression="lzf")

		self.images = self.f.create_dataset('images', (self.item_number, self.CHANNELS, self.img_size, self.img_size), dtype='uint8', chunks=(batch_size, self.CHANNELS, self.img_size, self.img_size))
		self.images.dims[0].label = self.CHANNEL_LABEL
		self.images.dims[1].label = self.HEIGHT_LABEL
		self.images.dims[2].label = self.WIDTH_LABEL
		#store images as strings
		#dt = h5py.special_dtype(vlen=np.dtype('uint8'))
		#self.images = self.f.create_dataset('images', (self.item_number, ), dtype=dt)
		#self.images.dims[0].label = "size"
		#self.images.dims[1].label = "str_value"
		if self.include_labels:
			self.targets = self.f.create_dataset('labels', (self.item_number, self.level_size), dtype='uint32', chunks=(batch_size, self.level_size))
			self.targets.dims[0].label = self.HIERARCHY_LABEL

	def close_HDF5(self):
		print("Closing HDF5 file...")
		#self.targets = self.targets.resize((self.item_number, self.level_size))
		#self.images = self.images.resize((self.item_number, self.CHANNELS, self.img_size, self.img_size))
		#print(self.images.shape, self.targets.shape)
		self.f.flush()
		self.f.close()
		print(self.item_number)
		#print(len(self.hierarchy['levels']['Family']), len(self.hierarchy['levels']['Genus']), len(self.hierarchy['levels']['Species']))
