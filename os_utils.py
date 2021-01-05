import os
from os import listdir

def get_files(fold):
    files = [f for f in listdir(fold) if os.path.isfile(os.path.join(fold, f))]
    return files
