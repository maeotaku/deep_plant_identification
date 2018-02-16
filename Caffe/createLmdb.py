import numpy as np
import random
import caffe
import sys
import datetime
import mysql.connector
import lmdb
from PIL import Image
import os
import math
import shutil
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

def writeMapping(train):    

    output = open("./mapping.txt",'w')
    print "--------"
    for i in species:
        s = str(species.index(i))+" --- "+i+"\n"
        output.write(s)
    print "--------"
    output.close()
    sys.exit(1)


    
def deleteEverything(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def writeLMDBS(inputs):

    # DELETE DIRS
    deleteEverything("./DATA/")
    deleteEverything("./LABEL/")
    oldtot =  len(inputs)

    print "Testing images"

    for a,b in enumerate(inputs):
        path= "/media/champ/31e58032-c984-4eaf-8467-ed8a49839cf7/DATA/LifeCLEF2015/Data Augmentation/2015-ALL/trainAndVal-256/"+b[1]+".jpg"
        im= np.array(Image.open(path)) # or load whatever ndarray you need
    
        if(im.shape==(256,256,3)):
            if(a%10000==0):
                print a
        else:
            inputs.pop(a)
            print "????",im.shape

    tot=len(inputs)
    print oldtot,tot





    print "Writing labels"
    d = os.path.join(dest,"LABEL")
    in_db = lmdb.open(d, map_size=int(1e12))
    cpt=0
    with in_db.begin(write=True) as in_txn:
        for in_idx, t in enumerate(inputs):
            if(cpt%1000==0):
                print "Label : %s / %s" % (cpt,tot)

            s = t[4]
            g = t[3]
            f = t[2]
            if s not in species:
                species.append(s)


            if g not in genus:
                genus.append(g)


            if f not in family:
                family.append(f)

            g = genus.index(g)
            f = family.index(f)
            s = species.index(s)
            
            im = np.zeros((3,1,1))
            im[0][0][0] = s # espece
            im[1][0][0] = g # genre
            im[2][0][0] = f # famille

            print s,g,f
            #print im
            #print "---"
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            cpt+=1
    in_db.close()



    
    

    print "Writing images"
    d = os.path.join(dest,"DATA")

    cpt=0
    for idx in range(int(math.ceil(len(inputs)/1000.0))):
        print (idx*1000),len(inputs)
        in_db_label = lmdb.open(d, map_size=int(1e12))
        with in_db_label.begin(write=True) as in_txn:
            print ((idx*1000),(idx*1000+1000)),len(inputs)
            for in_idx, in_ in enumerate(inputs[(idx*1000):(idx*1000+1000)]):
                path= "/media/champ/31e58032-c984-4eaf-8467-ed8a49839cf7/DATA/LifeCLEF2015/Data Augmentation/2015-ALL/trainAndVal-256/"+in_[1]+".jpg"
                if(cpt%1000==0):
                    print "Image : %s / %s" % (cpt,tot)
                

                im = np.array(Image.open(path)) # or load whatever ndarray you need

                if(im.shape==(256,256,3)):
                    im = im[:,:,::-1]
                    im = im.transpose((2,0,1))
                    im_dat = caffe.io.array_to_datum(im)
                    in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())
                    cpt+=1
                else:
                    print path
                    print "????",im.shape
                    
        in_db_label.close()







def oldie():
    
    in_db = lmdb.open(d, map_size=int(1e12))
    cpt=0
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            path= "/media/champ/31e58032-c984-4eaf-8467-ed8a49839cf7/DATA/LifeCLEF2015/Data Augmentation/2015-ALL/trainAndVal-256/"+in_[1]+".jpg"

            if(cpt%1000==0):
                print "Data : %s / %s" % (cpt,tot)


            im = np.array(Image.open(path)) # or load whatever ndarray you need

            if(im.shape==(256,256,3)):
                im = im[:,:,::-1]
                im = im.transpose((2,0,1))
                im_dat = caffe.io.array_to_datum(im)
                in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            else:
                print path
                print "????",im.shape
            cpt+=1
    in_db.close()









def getAll():
    res = []
    cnx = mysql.connector.connect(user='plantnet', database='plantnet', password='gxk1tTPfingBUh3r8I6xPYtM=', host='otmedia.lirmm.fr')
    cursor = cnx.cursor()

    query =  ("SELECT id,famille,genre, nom_retenu,num_observation FROM `lifeclef2015` ORDER BY RAND(123) LIMIT "+str(limit))
    
    print query
    cursor.execute(query)
  
    for (id,famille, genre, espece,num_observation) in cursor:
        n = (num_observation,id,famille,genre,espece)
        #print n
        res.append(n)
    cursor.close()
    cnx.close()

    

    return res













dest = "/home/champ/apps/caffeGPU/data/TEST-MULTI-LMDB"
species = []
genus = []
family = []
random.seed(0)
limit = 40000000
percTrain=100


cnx = mysql.connector.connect(user='plantnet', database='plantnet', password='gxk1tTPfingBUh3r8I6xPYtM=', host='otmedia.lirmm.fr')
cursor = cnx.cursor()

query =  "SELECT distinct num_observation FROM `lifeclef2015` ORDER BY RAND(123)"

print query
cursor.execute(query)

observations=[]


for (num_observation) in cursor:
    observations.append(num_observation[0])

cursor.close()
cnx.close()

print len(observations),"observations"

mmax = len(observations)*percTrain/100
trainObs= observations[:mmax]
print "Train :",0,mmax
print "Val :",mmax,len(observations)
valObs= observations[mmax:]



basePath="/home/champ/DEV/DatastoreCrawler/GUYANE-092015/gt/"


# RECUPERER TOUTE LA VERITE DE TERRAIN
gt = getAll()
nbImages = len(gt)
print nbImages,"images"




#PLACER CHAQUE IMAGE DANS LE TRAIN OU LE VAL
train = []
val = []
for e in gt:
    if(e[0] in trainObs):
        #print "train",e[0]
        train.append(e)
    if(e[0] in valObs):
        #print "val",e[0]
        val.append(e)


print "Train images : ",len(train)
writeLMDBS(train)
print "Val images : ",len(val)


writeMapping(train)



