from Model import *

from ModelGoogleNet import *

class GoogLeNetBaseline(GoogLeNet):

    @classmethod
    def build_for_transfer(cls, input_var, param_filename, organ_param_filename, organ_classes):

        #net = cls.build(input_var)
        #network  = cls.load_model_parameters_npz(organ_param_filename, net['prob'])
        _, organ_net = GoogLeNet.build_for_resume(input_var, organ_param_filename, organ_classes)
        organ_net = GoogLeNet.change_layer_names(organ_net, "organs")
        organ_net = GoogLeNet.make_untrainable(organ_net)

        net = cls.build(input_var)
        network  = cls.load_model_parameters(param_filename, net['prob'])


        del net['loss3/classifier']
        del net['prob']

        del organ_net['organs/loss3/classifier']
        del organ_net['organs/prob']

        net['concat_organs_and_more'] = ConcatLayer( [net['pool5/7x7_s1'], organ_net['organs/pool5/7x7_s1']] )

        net['loss3/classifier'] = DenseLayer(net['concat_organs_and_more'],
                                             num_units=NEW_TOTAL_CLASSES,
                                             nonlinearity=lasagne.nonlinearities.rectify)
        net['prob'] = NonlinearityLayer(net['loss3/classifier'], nonlinearity=softmax)

        network = net['prob']
        return network, net

    @classmethod
    def build_for_resume(cls, input_var, param_filename):
        pass
