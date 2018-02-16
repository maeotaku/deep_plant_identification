from Model import *

class VGG_S(Model):

    @classmethod
    def build(cls, input_var):
        net = {}
        net['input'] = InputLayer((None, 3, RESIZE_SIZE, RESIZE_SIZE), input_var = input_var)
        net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)#, flip_filters=False)
        net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
        net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3)#, ignore_border=False)
        net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)#, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2)#, ignore_border=False)
        net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)#, flip_filters=False)
        net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)#, flip_filters=False)
        net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)#, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3)#, ignore_border=False)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
        net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(net['drop7'], num_units=TOTAL_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
        return net

    @classmethod
    def load_model_parameters(cls, filename, network):
        saved_parameters = pickle.load(open(filename))
        values = saved_parameters['values']
        classes = saved_parameters['synset words']
        mean_image = saved_parameters['mean image']
        lasagne.layers.set_all_param_values(network, values)
        return network

    @classmethod
    def build_for_transfer(cls, input_var, param_filename):
        net = cls.build(input_var)
        network  = cls.load_model_parameters(param_filename, net['fc8'])
        if TOTAL_CLASSES != NEW_TOTAL_CLASSES:
            del net['fc8']
            net['fc8'] = DenseLayer(net['drop7'], num_units=NEW_TOTAL_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
        network = net['fc8']
        return network
