from TaggedImage import *

def get_new_uuid():
    return str(uuid.uuid1())

class IPlantDataSet(object):

    def __init__(self, destination_path):
        self.destination_path = destination_path

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError( "next not implemented" )
