from Model import *

class GoogLeNetHierarchical(Model):

    @classmethod
    def build_inception_module(cls, name, input_layer, nfilters, target=None):
        # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
        net = {}
        if RUN_MODE=='gpu':
            if not target is None:
                net['pool'] = PoolLayer(input_layer, pool_size=3, stride=1, pad=1)#, ignore_border=False)
                net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
                net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
                net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
                net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
                net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
                net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False, W=dev_var(lasagne.init.GlorotUniform('relu'), target), b=dev_var(lasagne.init.Constant(0.0), target))
            else:
                net['pool'] = PoolLayer(input_layer, pool_size=3, stride=1, pad=1)#, ignore_border=False)
                net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1, flip_filters=False)
                net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)
                net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, flip_filters=False)
                net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)
                net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, flip_filters=False)
                net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)
        else:
            net['pool'] = PoolLayer(input_layer, pool_size=3, stride=1, pad=1)#, ignore_border=False)
            net['pool_proj'] = ConvLayer(
                net['pool'], nfilters[0], 1)

            net['1x1'] = ConvLayer(input_layer, nfilters[1], 1)

            net['3x3_reduce'] = ConvLayer(
                input_layer, nfilters[2], 1)
            net['3x3'] = ConvLayer(
                net['3x3_reduce'], nfilters[3], 3, pad=1)

            net['5x5_reduce'] = ConvLayer(
                input_layer, nfilters[4], 1)
            net['5x5'] = ConvLayer(
                net['5x5_reduce'], nfilters[5], 5, pad=2)

        net['output'] = ConcatLayer([
            net['1x1'],
            net['3x3'],
            net['5x5'],
            net['pool_proj'],
            ])

        return {'{}/{}'.format(name, k): v for k, v in net.items()}

    @classmethod
    def build(cls, input_var):
        net = {}
        net['input'] = InputLayer(shape=(None, 3, IMG_SIZE, IMG_SIZE), input_var = input_var)
        if RUN_MODE=='gpu':
            net['conv1/7x7_s2'] = ConvLayer(
                net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
            net['pool1/3x3_s2'] = PoolLayer(
                net['conv1/7x7_s2'], pool_size=3, stride=2)#, ignore_border=False)
            net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
            net['conv2/3x3_reduce'] = ConvLayer(
                net['pool1/norm1'], 64, 1, flip_filters=False)
            net['conv2/3x3'] = ConvLayer(
                net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
            net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
            net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

        else: #CPU
            net['conv1/7x7_s2'] = ConvLayer(
                net['input'], 64, 7, stride=2, pad=3)
            net['pool1/3x3_s2'] = PoolLayer(
                net['conv1/7x7_s2'], pool_size=3, stride=2)#, ignore_border=False)
            net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
            net['conv2/3x3_reduce'] = ConvLayer(
                net['pool1/norm1'], 64, 1)
            net['conv2/3x3'] = ConvLayer(
                net['conv2/3x3_reduce'], 192, 3, pad=1)
            net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
            net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2)#, ignore_border=False)

        net.update(cls.build_inception_module('inception_3a',
                                          net['pool2/3x3_s2'],
                                          [32, 64, 96, 128, 16, 32]))
        net.update(cls.build_inception_module('inception_3b',
                                          net['inception_3a/output'],
                                          [64, 128, 128, 192, 32, 96]))
        net['pool3/3x3_s2'] = PoolLayer(
          net['inception_3b/output'], pool_size=3, stride=2)#, ignore_border=False)

        net.update(cls.build_inception_module('inception_4a',
                                          net['pool3/3x3_s2'],
                                          [64, 192, 96, 208, 16, 48]))
        net.update(cls.build_inception_module('inception_4b',
                                          net['inception_4a/output'],
                                          [64, 160, 112, 224, 24, 64]))
        net.update(cls.build_inception_module('inception_4c',
                                          net['inception_4b/output'],
                                          [64, 128, 128, 256, 24, 64]))
        net.update(cls.build_inception_module('inception_4d',
                                          net['inception_4c/output'],
                                          [64, 112, 144, 288, 32, 64]))
        net.update(cls.build_inception_module('inception_4e',
                                          net['inception_4d/output'],
                                          [128, 256, 160, 320, 32, 128]))
        net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'], pool_size=3, stride=2)#, ignore_border=False)

        net.update(cls.build_inception_module('inception_5a',
                                          net['pool4/3x3_s2'],
                                          [128, 256, 160, 320, 32, 128]))
        net.update(cls.build_inception_module('inception_5b',
                                          net['inception_5a/output'],
                                          [128, 384, 192, 384, 48, 128]))

        net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
        net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                             num_units=TOTAL_CLASSES,
                                             nonlinearity=linear
                                             #W=dev_var(lasagne.init.Glorot('relu'), 'dev1'),
                                             #b=dev_var(lasagne.init.Constant(0.0), 'dev1')
                                             )
        net['prob'] = NonlinearityLayer(net['loss3/classifier'], nonlinearity=softmax)
        return net

    @classmethod
    def load_model_parameters(cls, filename, network):
        saved_parameters = pickle.load(open(filename))
        lasagne.layers.set_all_param_values(network, saved_parameters)
        return network

    @classmethod
    def build_for_transfer(cls, input_var, param_filename):
        net = cls.build(input_var)
        network  = cls.load_model_parameters(param_filename, net['prob'])

        if TOTAL_CLASSES != NEW_TOTAL_CLASSES:
            del net['loss3/classifier']
            net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                                 num_units=NEW_TOTAL_CLASSES,
                                                 nonlinearity=lasagne.nonlinearities.rectify)
            del net['prob']
            net['probSpecies'] = NonlinearityLayer(net['loss3/classifier'], nonlinearity=softmax)
            net['probGenus'] = NonlinearityLayer(net['loss3/classifier'], nonlinearity=softmax)

            return net['probSpecies'], net['probGenus']
        else:
            return net['prob']
