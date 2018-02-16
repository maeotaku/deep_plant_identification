

from __future__ import print_function

import sys
import os
import time
import pickle
import urllib
import io
#import skimage.transform
import threading

from theano import printing


import numpy as np
import theano

print(theano.__version__)

import theano.tensor as T
import theano.tensor.extra_ops as Teops
import lasagne

from config import *
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax

import scipy.misc

from Visualizations import *
from DataAugmentation import *
from DataSetsManagement import *


#from Models.ModelVGG_S import *
from Models.ModelGoogleNet import *
#from Models.ModelGoogleNetLSTM import *
from Models.ModelGoogleNetSeveralFullyConnected import *
#from Models.ModelInception3 import *
#from Models.ModelResNet import *
from Logger import *

from Layers.HierarchicalCategoricalCrossentropy import *

from HierarchyManagement import *

from scipy import misc

#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

#logger = DLLogger("http://" + PASTA_LOG_SERVER + ":8120", "Theano Testing")


def save_model_parameters(filename, network):
	np.savez(filename, *lasagne.layers.get_all_param_values(network))

def train_from_scratch():
	pass

def predict():
	pass

def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def test_files(hdf5_filename):
	f, X, y = open_hdf5_file(hdf5_filename)
	separated_indexes = generate_separated_indexes(X, y, { "train": 0.8, "test" : 0.2 })#, "validation" : 0.2 })
	train_indexes = separated_indexes["train"]
	test_indexes = separated_indexes["test"]
	validation_indexes = separated_indexes["validation"]

	print(test_indexes)

def show_progress(msg, current, show_module_percentage):
	if current % show_module_percentage == 0:
		print(msg, current)

def load_mean_stdev(filename):
	stats = load_pickle(filename)
	return stats["mean"], stats["stdev"]

def load_model_for_resume(model, input_var, params_input_filename):
	return model.build_for_resume(input_var, params_input_filename)

def load_model_for_transfer_learning(model, input_var, params_input_filename):
	#return model.add_lstm(input_var, params_input_filename)
	return model.build_for_transfer(input_var, params_input_filename)

def log_training(epoch, iteration_progress, training_loss, validation_loss, accs , accs5):
	log_args = [ epoch + 1, iteration_progress, training_loss, validation_loss] + accs + accs5
	log_append(LOG_FILE, log_args)

def log_per_class_accuracy(epoch, iteration_progress, class_accuracies, classes_level_list):
	idx = 0
	for class_i in class_accuracies:
		log_args = [ epoch + 1, iteration_progress, idx, classes_level_list[idx], class_i]
		log_append(CLASS_LOG_FILE, log_args)
		idx+=1

'''
def calc_accuracy(prediction, target, k):
    return T.mean(T.any(T.eq(np.argsort(prediction, axis=1)[:, -k:], target.dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX)

def calc_accuracy_5(prediction, target):
    return calc_accuracy(prediction, target, 5)

def calc_accuracy_1(prediction, target):
    return calc_accuracy(prediction, target, 1)
'''

def transfer_train(train_filename, test_filename, params_output_filename, train_mean_filename="train_mean_file.pickle", test_mean_filename="test_mean_file.pickle", num_epochs=500, only_train=False):
	print("Loading data...")

	log_titles(LOG_FILE, ("Epoch", "Training Loss", "Validation Loss", "Validation Accuracy"))
	f_train, X, y = open_hdf5_file(train_filename)
	index_groups =  generate_separated_indexes(X, y, { "train" : 0.8, "val" : 0.2} )
	if not only_train:
		f_test, X_test, y_test = open_hdf5_file(test_filename)
		show_data_shapes(X, y, X, y, X_test, y_test)
	else:
		show_data_shapes(X, y, X, y, X, y)



	shape = (CHANNELS, RESIZE_SIZE, RESIZE_SIZE)

	train_mean, train_stdev = load_mean_stdev(train_mean_filename)
	train_mean, train_stdev = resize_mean_stdev(train_mean, train_stdev, IMG_SIZE, RESIZE_SIZE, CHANNELS)
	if not only_train:
		test_mean, test_stdev = load_mean_stdev(test_mean_filename)
		test_mean, test_stdev = resize_mean_stdev(test_mean, test_stdev, IMG_SIZE, RESIZE_SIZE, CHANNELS)

	#load hierarchy files
	#levels = ["Species", "Genus", "Family"]
	#levels = ["Class"]
	levels = ["Genus"]
	#levels = ["Family"]
	level_idxs = [1]
	n_levels = len(levels)

	classes_level_list, hierarchy, inv_hierarchy = initialize_hierarchies_from_file(HIERARCHIES_FILENAME, levels)
	n_classes_per_level = []
	for i in range(0, n_levels):
		n_classes_per_level += [ len(classes_level_list[levels[i]]) ]
		print(len(classes_level_list[levels[i]]))

	print(classes_level_list)
	#print(hierarchy)
	#print(inv_hierarchy)
	#print(classes_level_list)

	print("Building model and compiling functions...")
	input_var = T.tensor4('inputs') #32x3x256x256
	target_vars = [ T.ivector('targets' + str(i+1)) for i in range(0, n_levels) ]

	print("Setting up transfer learning...")
	networks = load_model_for_transfer_learning(ModelGoogleNetSeveralFullyConnected, input_var, INIT_WEIGHTS)
	predictions = lasagne.layers.get_output(networks)


	losses = []
	means = np.zeros(n_levels, dtype=np.float32)
	loss = 0
	for i in range(0, n_levels):
		losses += [ lasagne.objectives.categorical_crossentropy(predictions[i], target_vars[i]) ]
		loss = T.maximum(loss, losses[i].mean())
		#print(loss, loss.shape)
		#means[i] = loss
	#loss = np.amin(means)

	#loss = hierarchical_categorical_crossentropy(prediction, target_var, hierarchy, inv_hierarchy, levels)
	params = lasagne.layers.get_all_params(networks, trainable=True)
	learning_rate = T.scalar(name='learning_rate')
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=MOMENTUM)
	#updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
	#weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
	#loss += WEIGHT_DECAY * weightsl2
	test_predictions = lasagne.layers.get_output(networks, deterministic=True)
	test_losses = []
	test_means = np.zeros(n_levels, dtype=np.float32)
	test_loss = 0
	for i in range(0, n_levels):
		test_losses += [ lasagne.objectives.categorical_crossentropy(test_predictions[i], target_vars[i]) ]
		#test_loss = test_losses[i].mean()
		#test_means[i] = test_loss
		test_loss = T.maximum(test_loss, test_losses[i].mean())
	#test_loss = np.amin(test_means)
	#test_loss = test_loss / float(n_levels)

	test_accs = []
	test_accs5 = []
	test_class_accs = []
	for i in range(0, n_levels):
		test_accs += [ T.mean(T.eq(np.argsort(test_predictions[i], axis=1)[:, -1:], target_vars[i].dimshuffle(0, 'x')), dtype="float32") ]
		test_accs5 += [ T.mean(T.any(T.eq(np.argsort(test_predictions[i], axis=1)[:, -5:], target_vars[i].dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX) ]
		test_class_accs += [ T.mean(T.any(T.eq(np.argsort(test_predictions[i], axis=1)[:, -5:], target_vars[i].dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX) ]
		#[ np.true_divide(Teops.bincount( T.flatten(np.argsort(test_predictions[i], axis=1)[:, -1:])[T.eq(T.flatten(np.argsort(test_predictions[i], axis=1)[:, -1:]),  T.flatten(target_vars[i].dimshuffle(0, 'x'))).nonzero()], minlength=test_predictions[i].shape[1]), Teops.bincount(T.flatten(target_vars[i].dimshuffle(0, 'x')), minlength=test_predictions[i].shape[1])) ]
	args_fn = [input_var, learning_rate] + target_vars
	train_fn = theano.function(inputs=args_fn, outputs=loss, updates=updates)
	in_args_val = [input_var] + target_vars
	out_args_val = [test_loss] + test_accs + test_accs5 + test_class_accs
	val_fn = theano.function(inputs=in_args_val, outputs=out_args_val)

	def validation(n_levels, epoch, iteration_progress, X, y, train_mean, train_stdev, index_groups, train_batches, train_err):
		val_err = 0
		val_batches = 0
		val_batch_queue = deque()
		val_filler = threading.Thread(target=fill_batches_queue, args= (val_batch_queue, X, y, BATCH_SIZE, VAL_ITERATION_SIZE), kwargs={ 'img_size':RESIZE_SIZE, 'normalize':BATCH_NORMALIZE, 'mean':train_mean, 'stdev':train_stdev, 'original_indexes':index_groups['val']} )
		val_filler.start()
		val_accs = n_levels * [0.0]
		val_accs5 = n_levels * [0.0]
		val_class_accs = []
		for i in range(0, n_levels):
			val_class_accs += [ n_classes_per_level[i] * [0.0] ]
		for inputs, targets in queued_mini_batch_iterate(val_batch_queue, VAL_ITERATION_SIZE):
			it_start_time = time.time();
			try:
				#args = ( targets[:, i] for i in range(0, n_levels) )
				args = ( targets[:, i] for i in level_idxs )

				rest = val_fn(inputs, *args)
				err = rest[0]
				accs = rest[1:n_levels+1]
				accs5 = rest[n_levels+1: 2*n_levels + 1]
				class_accs = rest[2*n_levels + 1:]
				val_err += err
				for i in range(0, n_levels):
					#print(accs[i])
					val_accs[i] += accs[i]
					val_accs5[i] += accs5[i]
					#print(np.nan_to_num(class_accs[i]))
					val_class_accs[i] += np.nan_to_num(class_accs[i])
			except Exception as e:
				print(e)
			val_batches += 1
			print("Validation Iteration {} of {} in {:.3f}s".format(val_batches, VAL_ITERATION_SIZE, time.time() - it_start_time))

		training_loss = train_err / train_batches
		print("  training loss:\t\t{:.6f}".format(training_loss))
		validation_loss = val_err / val_batches
		print("  validation loss:\t\t{:.6f}".format(validation_loss))
		for i in range(0, n_levels):
			val_accs[i] = float(val_accs[i]) / float(val_batches) * 100.0
			val_accs5[i] = float(val_accs5[i]) / float(val_batches) * 100.0
			#val_class_accs[i] = (val_class_accs[i] / float(val_batches)) * 100.0
			#log_per_class_accuracy(epoch, iteration_progress, val_class_accs[i], classes_level_list[levels[i]])
		#print(val_class_accs)
		print("validation accuracy species: top-1 ".join(" \t\t{:.2f} %".format(k) for k in val_accs) + " top-5 ".join("\t\t{:.2f} %".format(k) for k in val_accs5))
		log_training(epoch + 1, iteration_progress, training_loss, validation_loss, val_accs, val_accs5)



	print("Starting training...")
	base_lr = BASE_LEARNING_RATE
	lr_decay = WEIGHT_DECAY
	iteration_progress = 0
	for epoch in range(num_epochs):
		train_err = 0
		train_batches = 0
		start_time = time.time()

		lr = base_lr #* lr_decay ** epoch


		batch_queue = deque()
		filler = threading.Thread(target=fill_batches_queue, args= (batch_queue, X, y, BATCH_SIZE, TRAIN_ITERATION_SIZE), kwargs={ 'img_size':RESIZE_SIZE, 'normalize':BATCH_NORMALIZE, 'mean':train_mean, 'stdev':train_stdev, 'original_indexes':index_groups['train']} )
		filler.start()
		for inputs, targets in queued_mini_batch_iterate(batch_queue, TRAIN_ITERATION_SIZE):
			it_start_time = time.time();
			#args = ( targets[:, i] for i in range(0, n_levels) ) #-1, -1, -1) )
			args = ( targets[:, i] for i in level_idxs ) #-1, -1, -1) )

			train_err += train_fn(inputs, lr, *args )
			train_batches += 1
			iteration_progress+=1
			print("Training Iteration {} of {} in {:.3f}s, queue has {} items left".format(iteration_progress, TRAIN_ITERATION_SIZE, time.time() - it_start_time, len(batch_queue)))
			if iteration_progress % int(VAL_EACH_X_ITERATIONS) == 0:
				validation(n_levels, epoch, iteration_progress, X, y, train_mean, train_stdev, index_groups, train_batches, train_err)
				#print("Saving parameters...")
				#save_model_parameters(params_output_filename.format(get_current_time_file_name()), network)

		print("Total Train Iterations: " , iteration_progress)
		#validate on epoch finish

		print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
		validation(n_levels, epoch, iteration_progress, X, y, train_mean, train_stdev, index_groups, train_batches, train_err)

	save_model_parameters(PARAMS_FILENAME, networks[0])


	# And a full pass over the test data:
	test_accs = n_levels * [0.0]
	test_accs5 = n_levels * [0.0]
	test_err = 0
	test_batches = 0
	test_batch_queue = deque()
	test_filler = threading.Thread(target=fill_batches_queue, args= (test_batch_queue, X_test, y_test, BATCH_SIZE, TEST_ITERATION_SIZE), kwargs={ 'img_size':RESIZE_SIZE, 'normalize':BATCH_NORMALIZE, 'mean':test_mean, 'stdev':test_stdev} )
	test_filler.start()
	for inputs, targets in queued_mini_batch_iterate(test_batch_queue, TEST_ITERATION_SIZE):

		args = ( targets[:, i] for i in range(0, n_levels) ) #-1, -1, -1) )
		rest = val_fn(inputs, *args)
		err = rest[0]
		accs = rest[1:n_levels+1]
		accs5 = rest[n_levels+1: ]
		test_err += err
		for i in range(0, n_levels):
			test_accs[i] += accs[i]
			test_accs5[i] += accs5[i]
		test_batches += 1
		show_progress("Test Iterations: ", test_batches, TEST_ITERATION_SIZE * 0.50)

	test_loss = test_err / test_batches
	print("  test loss:\t\t{:.6f}".format(test_loss))
	#test_accuracy1 = test_acc1 / test_batches * 100
	#	test_accuracy2 = test_acc2 / test_batches * 100
	#print("  test accuracy:\t\t{:.2f} % \t\t{:.2f} %".format(test_accuracy1, test_accuracy2))
	for i in range(0, n_levels):
		test_accs[i] = test_accs[i] / test_batches * 100
		test_accs5[i] = test__accs5[i] / test_batches * 100
	#log_append(LOG_FILE, (test_loss, test_accs, test_accs5))

	print("Saving final parameters...")
	save_model_parameters(params_output_filename.format(get_current_time_file_name()), network)
	log_training(epoch + 1, iteration_progress, training_loss, test_loss, test_loss, test_accs, test_accs5)


transfer_train(TRAIN_FILENAME, TEST_FILENAME, PARAMS_FILENAME, TRAIN_MEAN_FILENAME, TEST_MEAN_FILENAME, num_epochs=EPOCHS, only_train=True)
