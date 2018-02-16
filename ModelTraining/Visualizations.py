#this module uses pastalog https://github.com/rewonc/pastalog
#brew install node
'''
pip install pastalog
sudo pastalog --install
sudo pastalog --serve 8120
# - Open up http://localhost:8120/ to see the server in action.

Then run the python script with sudo as well
'''

from random import randrange
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from pastalog import Log
import base64
import uuid

# get a UUID - URL safe, Base64
def get_a_uuid():
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.replace('=', '')

class DLLogger():

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.logger = Log(path, name)

    def log(self, name, value, step):
        try:
            self.logger.post(name, value, step)
        except Exception as e:
            print("Cannot log visualization on Pasta", e)


#static generic functions
def show_random_img(dataset):
    random_index = randrange(0,len(dataset))
    img = dataset[random_index]
    plt.imshow(img.transpose())
    plt.show()

def show_img(img):
    #print(img.shape)
    plt.imshow(img.transpose())
    plt.show()

def show_imgs(imgs):
    for img in imgs:
        plt.imshow(img.transpose())
        #plt.figure(i+1)
    plt.show()

def save_img(img):
    scipy.misc.imsave(get_a_uuid() + ".jpg", img)

'''
#from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet
from lasagne import nonlinearities

class PlotLosses(object):
    def __init__(self, figsize=(8,6)):
        plt.plot([], [])

    def __call__(self, nn, train_history):
        train_loss = np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])

        plt.gca().cla()
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="test")

        plt.legend()
        plt.draw()
'''
