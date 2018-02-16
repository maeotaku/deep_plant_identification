from Base.IPlantDataSet import *
from Base.TaggedImage import *
import csv
import os
import urllib
import uuid
import datetime

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
        self.coreid = 0
        self.goodQualityAccessURI = 1
        #print row

    def get_coreid(self):
        return self.get_attribute(self.coreid)

    def get_goodQualityAccessURI(self):
        return self.get_attribute(self.goodQualityAccessURI)

class OcurrenceRow(CustomRow):

    def __init__(self, row):
        super(OcurrenceRow, self).__init__(row)
        self.id = 0
        self.kingdom = 43
        self.phylum = 60
        self.order = 59
        self.family = 27
        self.scientificName = 64
        #self.scientificNameAuthorship =
        self.genus = 31
        self.specificEpithet = 65
        self.taxonRank = 70
        self.infraspecificEpithet = 39
        self.dateIdentified = 18
        self.recordedBy = 10

        self.eventDate = 18
        #self.occurrenceRemarks = 35
        #self.habitat = 36
        self.country = 15
        self.locality = 49
        self.institutionCode = 40
        self.iDigBioUUID = 72
        #self.decimalLatitude = 56
        #self.decimalLongitude = 57

    def get_id(self):
        return self.get_attribute(self.id)

    def get_iDigBioUUID(self):
        return self.get_attribute(self.iDigBioUUID)

    def get_kingdom(self):
        return self.get_attribute(self.kingdom)

    def get_phylum(self):
        return self.get_attribute(self.phylum)

    def get_order(self):
        return self.get_attribute(self.order)

    def get_family(self):
        return self.get_attribute(self.family)

    def get_scientificName(self):
        return self.get_attribute(self.scientificName)

    def get_genus(self):
        return self.get_attribute(self.genus)

    def get_institutionCode(self):
        return self.get_attribute(self.institutionCode)

    def get_specificEpithet(self):
        return self.get_attribute(self.specificEpithet)

    def get_taxonRank(self):
        return self.get_attribute(self.taxonRank)

    def get_infraspecificEpithet(self):
        return self.get_attribute(self.infraspecificEpithet)

    def get_scientificNameAuthorship(self):
        return self.get_attribute(self.scientificNameAuthorship)

    def get_dateIdentified(self):
        return self.get_attribute(self.dateIdentified)

    def get_recordedBy(self):
        return self.get_attribute(self.recordedBy)

    def get_locality(self):
        return self.get_attribute(self.locality)

    def get_decimalLatitude(self):
        return self.get_attribute(self.decimalLatitude)

    def get_decimalLongitude(self):
        return self.get_attribute(self.decimalLongitude)

    def get_event_date(self):
        return self.get_attribute(self.eventDate)

    def get_country(self):
        return self.get_attribute(self.country)

class Properties(dict):

    def _clean_species_name(self, text):
        try:
            idx = text.upper().find(" VAR.")
            if (idx!=-1):
                return text[:idx-1]
            return text
        except Exception as e:
            return "Uknown"

    def is_complete_species_name(self):
        if (len(self.species_name.split()) < 2):
            return False
        return True

    def get_uuid(self):
        return super(Properties,self).__getitem__("UUID")

    def _buil_date(self, row_occ):
        #dt = datetime.date(year=int(row_occ.get_year()),day=int(row_occ.get_day()),month=int(row_occ.get_month()))
        #return dt.strftime("%Y-%m-%d")
        return str(row_occ.get_year()) + "-" + str(row_occ.get_day()) + "-" + str(row_occ.get_month())


    def __init__(self, row_occ, row_img, *args, **kw):
        super(Properties,self).__init__(*args, **kw)
        super(Properties,self).__setitem__("Order", row_occ.get_order())
        super(Properties,self).__setitem__("Genus", row_occ.get_genus())
        super(Properties,self).__setitem__("Family", row_occ.get_family())
        if (row_occ.get_genus() is None or row_occ.get_specificEpithet() is None):
            self.species_name = self._clean_species_name(row_occ.get_scientificName())
        else:
            self.species_name = row_occ.get_genus() + " " + row_occ.get_specificEpithet()
        super(Properties,self).__setitem__("Species", self.species_name)
        super(Properties,self).__setitem__("Author", row_occ.get_recordedBy())
        super(Properties,self).__setitem__("Date", row_occ.get_event_date())
        #super(Properties,self).__setitem__("Latitude", row_occ.get_decimalLatitude())
        #super(Properties,self).__setitem__("Longitude", row_occ.get_decimalLongitude())
        super(Properties,self).__setitem__("Location", row_occ.get_locality())
        super(Properties,self).__setitem__("Country", row_occ.get_country())
        super(Properties,self).__setitem__("UUID", row_occ.get_iDigBioUUID())
        super(Properties,self).__setitem__("InstitutionCode", row_occ.get_institutionCode())

        super(Properties,self).__setitem__("URL", row_img.get_goodQualityAccessURI())

class IDigBioDataSet(IPlantDataSet):

    def __init__(self, destination_path, occurrence_file_path, images_file_path):
        super(IDigBioDataSet, self).__init__(destination_path)
        #IPlantDataSet.__init__(self, destination_path)

        self.occurrence_file_path = occurrence_file_path
        self.images_file_path = images_file_path

        print("Loading ocurrence file...")
        self.ocurrenciesDic = {}
        self.occurrence_data = self._read_csv_file(occurrence_file_path)
        self._gen_occurencies_dict()
        #print(self.ocurrenciesDic)
        print("Loading images file...")
        self.images_data = self._read_csv_file(images_file_path)
        #print("Loading identification file...")
        #self.identification_data = self.read_csv_file(identification_file_path)

        print("File loaded...", len(self.occurrence_data), len(self.images_data))#, len(self.identification_data))

        self.current = 0
        self.high = len(self.images_data)


    def _gen_occurencies_dict(self):
        for row in self.occurrence_data:
            occur = OcurrenceRow(row)
            try:
                self.ocurrenciesDic[occur.get_id()] = occur
            except Exception as e:
                print e

    def _read_csv_file(self, filename):
        data=[]
        with open(filename, "rb") as csvfile:
            datareader=csv.reader(csvfile)
            for row in datareader:
                data.append(row)
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
                idx = row_img.get_coreid()
                #print("Trying..." ,idx)
                if idx in self.ocurrenciesDic:
                    row_occ = self.ocurrenciesDic[idx]
                    #get rid of sub species stuff and keep only property identified images
                    props = Properties(row_occ, row_img)
                    if (props.is_complete_species_name()):
                        uuid = props.get_uuid()
                        #uuid = get_new_uuid()
                        return  self.create_new_img(uuid, row_img.get_goodQualityAccessURI(), props)
            raise StopIteration
