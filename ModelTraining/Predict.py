from __future__ import print_function

import sys
import os
import time
import pickle
import urllib
import io
import skimage.transform

import numpy as np
import theano
import theano.tensor as T
import lasagne

#from run_model import *
from config import *
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax

from scipy import misc
import scipy.misc

from DataSetsManagement import *
from Models.ModelGoogleNet import *
from Logger import *

def load_model_for_resume(model, input_var, params_input_filename):
    return model.build_for_resume(input_var, params_input_filename)

class Predictor():
	def __init__(self):
		self.levels = ["species"]
		self.classes_level_list, self.hierarchy, self.inv_hierarchy = initialize_hierarchies_from_file(HIERARCHIES_FILENAME, self.levels)
		print("Building model and compiling functions...")
		self.input_var = T.tensor4('inputs')
		self.target_var = T.ivector('targets')
		print("Loading weights...")
		self.network = load_model_for_resume(GoogLeNet, self.input_var, INIT_WEIGHTS)
		self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		self.try_fn = theano.function([self.input_var], [self.test_prediction], on_unused_input='ignore')
		print("Starting guess...")

	def reorder_img(self,img):
		try:
			img = np.swapaxes(img, 0, 1)
			img = np.swapaxes(img, 0, 2)
			return img
		except Exception as e:
			print("Cannot resize image", e)
			return None

	def insertion_sort(self, aList ,bList):
		for i in range( 0, len( aList ) ):
			tmp = aList[i]
			tmp2 = bList[i]
			k = i
			while k > 0 and tmp > aList[k - 1]:
				aList[k] = aList[k - 1]
				bList[k] = bList[k-1]
				k -= 1
			aList[k] = tmp
			bList[k]= tmp2
		return aList,bList

	def predict(self, img_path):
		inputs = misc.imread(img_path, mode='RGB')
		resized = misc.imresize(inputs, (224, 224, 3), 'bilinear', mode=None)
		resized=reorder_img(resized)
		#resized = theano.shared(inputs)
		#target=resized
		#guess = try_fn([resized,target])
		guess = self.try_fn([resized])
		lista_clases=guess[0].tolist()
		my_list = [i for i in range(len(lista_clases[0]))]
		lista_clases,my_list=self.insertion_sort(lista_clases[0],my_list)
		diccionario_clases = pickle.load( open( HIERARCHIES_FILENAME, "rb" ) )
		lista_genes=diccionario_clases['levels']['Family']
		print("Guess is: ")
		for i in range(TOP_GUESSES):
			print("#",i+1, lista_genes[my_list[i]])

prueba= Predictor()
prueba.predict("/Datasets/RAW/All_CR_Leaves_Cleaned/Thouinidium decandrum/Thouinidium decandrum_3_4_2.JPG")
