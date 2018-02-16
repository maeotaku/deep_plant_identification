from Base.IPlantDataSet import *
from Base.TaggedImage import *
import csv
import os
import urllib
import uuid
import datetime
import re
from xml.sax.saxutils import unescape

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

def get_file_list(a_dir):
    files = next(os.walk(a_dir))[2]
    return clean_hidden_files(files)

class CustomRow(object):

    def __init__(self, row):
        self.row = row

    def get_attribute(self, idx):
        try:
            value = self.row[idx]
            if value == "":
                return None
            return value
        except Exception as e:
            #print e
            return None


class ImageRow(CustomRow):

    def __init__(self, row):
        super(ImageRow, self).__init__(row)
        self.cropCommonName = 0
        self.cropScientificName = 1
        self.diseaseCommonName = 2
        self.diseaseScientificName = 3
        self.goodQualityAccessURI = 4
        self.description = 5
        self.metadata = 6
        #print row

    def strip_html(self, text):
        text = re.sub('[^A-Za-z0-9\s]+', '', text)
        return text

    def get_cropCommonName(self):
        return self.get_attribute(self.cropCommonName)

    def get_cropScientificName(self):
        text = self.get_attribute(self.cropScientificName)
        try:
            text = self.strip_html(text)
            return text
        except Exception as e:
            return ""


    def get_diseaseCommonName(self):
        return self.get_attribute(self.diseaseCommonName)

    def get_diseaseScientificName(self):
        text = self.get_attribute(self.diseaseScientificName)
        try:
            text = self.strip_html(text)
            return text
        except Exception as e:
            return ""

    def get_goodQualityAccessURI(self):
        return self.get_attribute(self.goodQualityAccessURI)

    def get_description(self):
        return self.get_attribute(self.description)


class Properties(dict):

    def get_uuid(self):
        return get_new_uuid()

    def __init__(self, row_img, *args, **kw):
        super(Properties,self).__init__(*args, **kw)
        super(Properties,self).__setitem__("Species", row_img.get_cropCommonName())
        super(Properties,self).__setitem__("CropCommonName", row_img.get_cropCommonName())
        super(Properties,self).__setitem__("CropScientificName", row_img.get_cropScientificName())
        super(Properties,self).__setitem__("DiseaseCommonName", row_img.get_diseaseCommonName())
        super(Properties,self).__setitem__("DiseaseScientificName", row_img.get_diseaseScientificName())
        super(Properties,self).__setitem__("URL", row_img.get_goodQualityAccessURI())
        super(Properties,self).__setitem__("Description", row_img.get_description())

class PlantVillageDataSet(IPlantDataSet):

    def __init__(self, destination_path, images_file_path):
        super(PlantVillageDataSet, self).__init__(destination_path)

        self.images_file_path = images_file_path

        print("Loading images file...")
        self.images_data = self._read_several_csv_files(images_file_path)

        print("File loaded...", len(self.images_data))

        self.current = 0
        self.high = len(self.images_data)

    def _read_csv_file(self, filename):
        data=[]
        with open(filename, "rb") as csvfile:
            datareader=csv.reader(csvfile)
            for row in datareader:
                data.append(row)
        return data

    def _read_several_csv_files(self, path):
        data=[]
        for filename in get_file_list(path):
            try:
                data += self._read_csv_file(os.path.join(path, filename))
            except Exception as e:
                raise
        return data

    def create_new_img(self, img_name, url, properties):
        return TaggedImageURL(img_name, url=url, properties=properties)

    def next(self):
        if self.current > self.high:
            raise StopIteration
        else:
            while self.current < self.high:
                self.current += 1
                row_img = ImageRow(self.images_data[self.current - 1])
                props = Properties(row_img)
                uuid = props.get_uuid()
                return  self.create_new_img(uuid, row_img.get_goodQualityAccessURI(), props)
            raise StopIteration
