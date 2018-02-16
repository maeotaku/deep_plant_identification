import os
import time

import cPickle

import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import csv
import shutil

import sys
import errno
from subprocess import call
import caffe

import json
import numpy

numpy.set_printoptions(threshold=numpy.nan)

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

CAFFE_MODEL_DEF_FILE='/home/goeau/EXPERIMENTATIONS/Data/caffe_models/PlantCLEF2015Model/models/deploy_new_style.prototxt'.format(REPO_DIRNAME))
CAFFE_PRETRAINED_MODEL_FILE='/home/goeau/EXPERIMENTATIONS/Data/caffe_models/PlantCLEF2015Model/models/2015-FINAL_iter_120000.caffemodel'.format(REPO_DIRNAME))
CAFFE_CLASS_LABELS_FILE='/home/goeau/EXPERIMENTATIONS/Data/caffe_models/PlantCLEF2015Model/mapping.txt'.format(REPO_DIRNAME))
CAFFE_MEAN_FILE=""

# Obtain the flask app object
app = flask.Flask(__name__)

image_dim = 256
raw_scale = 255.
gpu_mode = True #False#'True'

#classifierids = ['googlenet','plantclef2015']
#classifierids = ['plantclef2015','plantclef2015BNPReLU']
#classifierids = ['plantclef2016']
#classifierids = ['plantclef2016BatchNormPReLU']
#classifierids = ['googlenet']
#classifierids = ['plantclef2016Taxo','plantclef2016TaxoGenus','plantclef2016TaxoFamily','plantclef2016taxoflat']
classifierids = ['plantclef2016TaxoFlat']




def allowed_file(filename):
    _, extension = os.path.splitext(filename)
    return extension in ALLOWED_IMAGE_EXTENSIONS

#Can be used to utilize the same class ids for different datasets when doing transfer learning
def caffe_class_file_to_dictionary(filename):
    caffe_classes = {}
    inv_caffe_classes = {}
    with open(filename,'r') as f:
        for line in f:
            idx, name = line.split()
            caffe_classes[name] = idx
            inv_caffe_classes[idx] = name
    return caffe_classes, inv_caffe_classes

class CaffeClassifierExecutor(object):

    def __init__(self, name, labels_filename, model_filename, weights_filename, mean_filename):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.name = name
        self.net = {}
        self.labels, self.inv_labels = caffe_class_file_to_dictionary(labels_filename)
        self.load_classifier(model_filename, weights_filename, mean_filename)
        '''
        if 'googlenet' in ids:
            self.net['googlenet'] = caffe.Classifier(
                googlenet_args['CAFFE_MODEL_DEF_FILE'],googlenet_args['CAFFE_PRETRAINED_MODEL_FILE'],
                image_dims=(image_dim,image_dim), raw_scale= raw_scale,
                mean=np.load( googlenet_args['mean_file']).mean(1).mean(1), channel_swap=(2, 1, 0)
            )
            with open(googlenet_args['CAFFE_CLASS_LABELS_FILE']) as f:
                labels_df1 = pd.DataFrame([
                    {
                        'synset_id': l.strip().split(' ')[0],
                        'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                    }
                    for l in f.readlines()
                ])
                self.labels['googlenet'] = labels_df1.sort('synset_id')['name'].values

        if 'plantclef2015' in ids:
            self.load_classifier('plantclef2015', plantclef2015_args)

        if 'plantclef2016' in ids:
            self.load_classifier('plantclef2016', plantclef2016_args)

        if 'plantclef2015BNPReLU' in ids:
            self.load_classifier('plantclef2015BNPReLU', plantclef2015BNPReLU_args)

        if 'plantclef2016BatchNormPReLU' in ids:
            self.load_classifier('plantclef2016BatchNormPReLU', plantclef2016BatchNormPReLU_args)

        if 'plantclef2016Taxo' in ids:
            self.load_classifier('plantclef2016Taxo', plantclef2016Taxo_args)

        if 'plantclef2016TaxoGenus' in ids:
            self.load_classifier('plantclef2016TaxoGenus', plantclef2016TaxoGenus_args)

        if 'plantclef2016TaxoFamily' in ids:
            self.load_classifier('plantclef2016TaxoFamily', plantclef2016TaxoFamily_args)

        if 'plantclef2016TaxoFamily' in ids or 'plantclef2016TaxoGenus' in ids:
            self.load_taxonomy()

        if 'plantclef2016TaxoFlat' in ids:
            self.load_classifier('plantclef2016TaxoFlat', plantclef2016taxoflat_args)
        '''
        #load
        for c in self.net:
            self.net[c].forward()


    #def load_classifier(self, name, args):
    def load_classifier(self, name, model_filename, weights_filename, mean_filename):
        '''
        if 'mean' in args:
            mean = np.load(args['mean_file']).mean(1).mean(1)
        else:
            mean = np.load(googlenet_args['mean_file']).mean(1).mean(1)
        '''
        self.net[name] = caffe.Classifier(
            model_filename,
            weights_filename,
            image_dims=(image_dim,image_dim),
            raw_scale= raw_scale,
            mean=np.load(mean_filename).mean(1).mean(1),
            channel_swap=(2, 1, 0)
        )
        '''
        #be carefull ids plantnet will be sorted, must cast int
        with open(args['CAFFE_CLASS_LABELS_FILE']) as f:
            labels_df2 = pd.DataFrame([
                {
                    'name': l.strip().split(' ')[0],
                    'synset_id': int(' '.join(l.strip().split(' ')[1:]).split(',')[0])
                }
                for l in f.readlines()
            ])
        self.labels[name] = labels_df2.set_index('synset_id')['name'].to_dict()
        '''
    '''
    def get_labels(self, classifierid):
        try:
            labels_as_full_string_dic = {}
            ls = self.labels[classifierid]
            print type(ls)
            print len(ls)
            if type(ls) is dict:
                for i in ls:
                    #print(i, ' ', ls[i])
                    labels_as_full_string_dic[str(i)] = ls[i]
            elif type(ls) is np.ndarray:
                labellist = ls.tolist()
                for i in range(0,len(labellist)):
                    labels_as_full_string_dic[str(i)] = ls[i]
            return labels_as_full_string_dic
        except Exception as err:
            logging.info('get labels: %s', err)
            return (False, 'Something went wrong, no init?')
    '''


    def classify_image(self, image, responsesize, oversample, humanclassname):
        try:
            scores = self.net[self.name].predict([image], oversample).flatten()
            indices = (-scores).argsort()[:int(responsesize)]
            scoreswithlabels = {}
            results = {}
            results['scores'] = scoreswithlabels
            #print 'results ', results
            for i in indices:
                #print 'i ', i
                #if i in self.labels[classifierid]:
                if ("{:10.4f}".format(scores[i],5)) != "    0.0000":
                    #print 'self.labels[classifierid][i] ', self.labels[classifierid][i]
                    #print 'scores[i] ', scores[i]
                    scoreswithlabels[self.labels[classifierid][i]] = "{:10.4f}".format(scores[i],5).lstrip()
                else: break

            results['featuresvector'] = self.net[classifierid].blobs['pool5/7x7_s1'].data[0].reshape(1,-1).tolist()[0]

            return results

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

    '''
    def classify_images(self, urls, images, classifierid, responsesize, humanclassname):
        try:
            oversample=False
            #print 'images : ', images
            #print 'urls : ', urls
            resultss = {}
            scoress = self.net[classifierid].predict(images, oversample).flatten()
            #class_number = len(self.net[classifierid].blobs['loss3/classifier_PlantCLEF2016'].data[0].reshape(1,-1))
            class_number = 1000
            if classifierid == 'plantclef2016TaxoFlat':
                class_number = 10000
            #print class_number
            #print scoress
            for j in range(0,len(urls)):
                #print j
                print urls[j], ' ', j, ' / ', len(urls)
                start=j*class_number#1000
                end=start+class_number#1000
                #print start, ' ', end
                #print scoress[:j]
                #print type(scoress)
                scores = scoress[start:end]
                #print 'scores ', scores
                indices = (-scores).argsort()[:int(responsesize)]
                #print 'indices ', indices, ' , number of indices : ', len(indices)
                scoreswithlabels = {}
                results = {}
                results['scores'] = scoreswithlabels
                #print 'results ', results
                for i in indices:
                    #print 'i ', i, ' ', scores[i]
                    #if i in self.labels[classifierid]:
                    if ("{:10.4f}".format(scores[i],5)) != "    0.0000":
                        #print 'self.labels[classifierid][i] ', self.labels[classifierid][i]
                        #print 'scores[i] ', scores[i]
                        scoreswithlabels[self.labels[classifierid][i]] = "{:10.4f}".format(scores[i],5).lstrip()
                    else: break
                #print results['scores']
                results['featuresvector'] = self.net[classifierid].blobs['pool5/7x7_s1'].data[j].reshape(1,-1).tolist()[0]
                resultss[urls[j]] = results
                print urls[j], ' with results ', resultss[urls[j]]['scores']
            return resultss


        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying one of the '
                           'images. Maybe try another ones?')
        '''

    '''
    def load_taxonomy(self):
        self.c_to_g = {}
        self.c_to_f = {}
        with open('/opt/data_plantclef/PlantCLEF/PlantCLEF2015MasterFinal.csv', 'rb') as csvfile:
            data  = csv.reader(csvfile, delimiter=';', quotechar='"')
            data.next()
            for row in data:
                f = row[4]
                g = row[5]
                c = row[17]
                s = row[6]
                self.c_to_g[c] = g
                self.c_to_f[c] = f
                print c, ' -> ', f


    def classify_images_taxo(self, urls, images, responsesize, humanclassname,oversample):
        try:
            #oversample=True
            #print 'image : ', image
            resultss = {}
            scoress_species = self.net['plantclef2016Taxo'].predict(images, oversample).flatten()
            scoress_genus = self.net['plantclef2016TaxoGenus'].predict(images, oversample).flatten()
            scoress_family = self.net['plantclef2016TaxoFamily'].predict(images, oversample).flatten()

            for j in range(0,len(urls)):
                #print j
                #print urls[j], ' ', len(urls)
                start=j*1000
                end=start+1000
                #print start, ' ', end
                #print scoress[:j]
                #print type(scoress)
                scores_species = scoress_species[start:end]
                scores_genus = scoress_genus[start:end]
                scores_family = scoress_family[start:end]

                #print 'scores ', scores
                indices_species = (-scores_species).argsort()[:int(responsesize)]
                indices_genus = (-scores_genus).argsort()[:int(responsesize)]
                indices_family = (-scores_family).argsort()[:int(responsesize)]
                #print 'indices ', indices
                scoreswithlabels_species = {}
                scoreswithlabels_genus = {}
                scoreswithlabels_family = {}
                scoreswithlabels_species_revised_genus = {}
                scoreswithlabels = {}

                results = {}
                results['scores_species'] = scoreswithlabels_species
                results['scores_genus'] = scoreswithlabels_genus
                results['scores_family'] = scoreswithlabels_family
                results['scores_species_revised_genus'] = scoreswithlabels_species_revised_genus
                results['scores'] = scoreswithlabels

                for i in indices_species:
                    #print 'i ', i
                    #if i in self.labels[classifierid]:
                    if ("{:10.4f}".format(scores_species[i],5)) != "    0.0000":
                        #print 'self.labels[classifierid][i] ', self.labels[classifierid][i]
                        #print 'scores[i] ', scores[i]
                        scoreswithlabels_species[self.labels['plantclef2016Taxo'][i]] = "{:10.4f}".format(scores_species[i],5).lstrip()
                    else: break
                #print results['scores']

                for i in indices_genus:
                    if ("{:10.4f}".format(scores_genus[i],5)) != "    0.0000":
                        scoreswithlabels_genus[self.labels['plantclef2016TaxoGenus'][i]] = "{:10.4f}".format(scores_genus[i],5).lstrip()
                    else: break

                for i in indices_family:
                    if ("{:10.4f}".format(scores_family[i],5)) != "    0.0000":
                        scoreswithlabels_family[self.labels['plantclef2016TaxoFamily'][i]] = "{:10.4f}".format(scores_family[i],5).lstrip()
                    else: break

                scoreswithlabelstmpgenus = {}
                scoreswithlabelstmp = {}

                #revise species with genus

                for i in scoreswithlabels_species:
                    #print i
                    classid = i.replace('train/','')
                    #print classid
                    current_score_species = scoreswithlabels_species[i]
                    #print classid, ' ', current_score_species

                    g = self.c_to_g[classid]
                    current_score_genus = 0.0
                    for genus in scoreswithlabels_genus:
                        #print g, ' ==? ', genus
                        if genus == g:
                            current_score_genus = scoreswithlabels_genus[genus]
                            #print current_score_genus
                            break
                    #scoreswithlabelstmpgenus[i] = float(current_score_species)*float(current_score_genus)
                    scoreswithlabelstmpgenus[i] = float(current_score_species)+float(current_score_genus)

                    f = self.c_to_f[classid]
                    current_score_family  = 0.0
                    for family in scoreswithlabels_family:
                        #print f, ' ==? ', family
                        if family == f:
                            current_score_family = scoreswithlabels_family[family]
                            #print current_score_family
                            break
                    #scoreswithlabelstmp[i] = scoreswithlabelstmpgenus[i]*float(current_score_family)
                    scoreswithlabelstmp[i] = (scoreswithlabelstmpgenus[i]+float(current_score_family)) / 3.0

                for i in scoreswithlabelstmpgenus:
                    #score = float(scoreswithlabelstmpgenus[i])
                    score = float(scoreswithlabelstmpgenus[i])  / 2.0
                    if ("{:10.4f}".format(score,5)) != "    0.0000":
                        scoreswithlabels_species_revised_genus[i] = "{:10.4f}".format(score,5).lstrip()

                for i in scoreswithlabelstmp:
                    #no norm: risk to boost score of false positive
                    score = float(scoreswithlabelstmp[i])
                    #print 'score norm ', score
                    if ("{:10.4f}".format(score,5)) != "    0.0000":
                        scoreswithlabels[i] = "{:10.4f}".format(score,5).lstrip()

                    #scores_species[i]
                print urls[j]
                print results['scores']

                results['featuresvector'] = self.net['plantclef2016Taxo'].blobs['pool5/7x7_s1'].data[j].reshape(1,-1).tolist()[0]

                resultss[urls[j]] = results
            return resultss

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying one of the '
                           'images. Maybe try another ones?')
        '''

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    classifierid = flask.request.args.get('classifierid', '')
    responsesize = flask.request.args.get('responsesize', '')
    oversample = flask.request.args.get('oversample', '')
    humanlabels = flask.request.args.get('humanlabels', '')

    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        #return flask.render_template(
        #    'index.html', has_result=True,
        #    result=(False, 'Cannot open image from URL.')
        #)
        resp = {"msg" : "not reachable image"}
        return flask.jsonify(**resp)

    logging.info('Image: %s %s', classifierid, imageurl)
    result = app.clf.classify_image(image, classifierid, responsesize, oversample, humanlabels)
    #print result
    return flask.jsonify(**result)

@app.route('/classify_urls', methods=['GET'])
def classify_urls():
    imageurls = flask.request.args.get('imageurls', '')
    responsesize = flask.request.args.get('responsesize', '')
    oversample = flask.request.args.get('oversample', '')
    humanlabels = flask.request.args.get('humanlabels', '')

    urls = imageurls.split(',')
    try:
        images=[]
        for i in range(0,len(urls)):
            string_buffer = StringIO.StringIO(urllib.urlopen(urls[i]).read())
            images.append(caffe.io.load_image(string_buffer))
    except Exception as err:
        logging.info('URL Image open error: %s', err)
        resp = {"msg" : "not reachable image"}
        return flask.jsonify(**resp)

    logging.info('Images: %s %s', classifierid, imageurls)
    resultss = []
    resultss = app.clf.classify_images(urls, images, classifierid, responsesize, humanlabels)
    for i in range(0,len(urls)):
        print urls[i], ' -> ', resultss[urls[i]]['scores']
    return flask.jsonify(**resultss)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    parser = optparse.OptionParser()
#    parser.add_option(
#        '-c', '--config',
#        help="database configuration",
#        type="str",
#        dest="config_file",
#        default=False)
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()


    app.clf = CaffeClassifierExecutor(classifierids)
#    if opts.config_file != False:
#        print 'opts.config_file ', opts.config_file
#        app.clf.load_CaffeClassifierExecutor(opts.config_file)
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
