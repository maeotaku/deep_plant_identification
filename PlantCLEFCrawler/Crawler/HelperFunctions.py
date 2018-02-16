import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import re
import uuid

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
