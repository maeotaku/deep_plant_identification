import os
import io
import uuid
import urllib2
import xml.etree.cElementTree as ET
from shutil import copyfile
from PIL import Image, ImageFile
import PIL

class SpecialTags(object):
    Species = "Species"
    Filename = "FileName"
    Image = "Image"

class XMLTags(object):

    def __init__(self, properties, filename):
        root = ET.Element(SpecialTags.Image)
        for key in properties:
            value = properties[key]
            ET.SubElement(root, key).text = value
        ET.SubElement(root, SpecialTags.Filename).text = filename
        self.tree = ET.ElementTree(root)

    def save(self, path):
        self.tree.write(path)

class TaggedImage(object):

    def __init__(self, img_name=None, properties=None):
        self.properties = properties
        self.img_name = img_name
        self.img_name_xml = self.img_name + ".xml"
        self.img_name_raw = self.img_name + ".jpg"
        self.tagsXml = XMLTags(self.properties, self.img_name)

    def get_property_name(self, key):
        return str(self.properties[key])

    def save(self, path):
        '''
        print("Downloading image...", self.url)
        try:
            urllib.urlretrieve(self.url, os.path.join(path, self.img_name_raw))
            self.tagsXml.save(os.path.join(path, self.img_name_xml))
        except Exception as e:
            print("Failed to download image... ", e)
            pass
        '''
        pass

    def load(self, path):
        pass

class TaggedImageURL(TaggedImage):

    def __init__(self, img_name=None, properties=None, url=None, base_width=1024):
        super(TaggedImageURL, self).__init__(img_name, properties)
        self.url = url
        self.base_width = base_width

    def save(self, path):
        try:
            img_path = os.path.join(path, self.img_name_raw)
            if not os.path.exists(img_path):
                print("Downloading image...", self.url)
                request = urllib2.urlopen(self.url, timeout=500)
                img = Image.open(io.BytesIO(request.read()))

                w = img.size[0]
                if self.base_width > w:
                    self.base_width = w
                w_percent = (self.base_width/float(w))
                h_size = int((float(img.size[1])*float(w_percent)))
                img = img.resize((self.base_width, h_size), PIL.Image.ANTIALIAS)

                img.save(img_path, quality=90, optimize=True)
            self.tagsXml.save(os.path.join(path, self.img_name_xml))
        except Exception as e:
            print("Failed to download image... ", e)
            pass

class TaggedImageLocalFile(TaggedImage):

    def __init__(self, img_name=None, properties=None, local_filename=None):
        super(TaggedImageLocalFile, self).__init__(img_name, properties)
        self.local_filename = local_filename

    def save(self, path):
        print("Copying image...", self.local_filename)
        try:
            copyfile(self.local_filename, os.path.join(path, self.img_name_raw))
            self.tagsXml.save(os.path.join(path, self.img_name_xml))
        except Exception as e:
            print("Failed to copy image and save XML...", e)
            pass
