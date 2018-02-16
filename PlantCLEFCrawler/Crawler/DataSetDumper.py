import os
import threading

import HelperFunctions as hf

def get_new_uuid():
    return str(uuid.uuid1())

class DataSetDumper(object):

        def __init__(self, dataset):
            self.dataset = dataset
            self.threads = []
            self.MAX_THREADS = 3

        def _add_folder(self, path):
            hf.create_folder(path)

        def _add_tagged_img(self, img, path):
            func = img.save
            self.launch_thread(func, (path,))

        def launch_thread(self, func, func_args):
            t = threading.Thread(target=func, args=func_args)
            t.start()
            self.threads+= [ t ]
            if (len(self.threads) == self.MAX_THREADS):
                for t in self.threads:
                    t.join()
                self.threads = []

        def generate(self):
            hf.create_folder(self.dataset.destination_path)
            for img in self.dataset:
                if not img is None:
                    path = os.path.join(self.dataset.destination_path, img.get_property_name("Species"))
                    self._add_folder(path)
                    self._add_tagged_img(img, path)
