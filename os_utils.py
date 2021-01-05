import os
import yaml
from os import listdir

def get_files(fold):
    files = [f for f in listdir(fold) if os.path.isfile(os.path.join(fold, f))]
    return files

# yaml configuration file operation class
class Params:
    def __init__(self, yaml_file):
        self.params = yaml.safe_load(open(yaml_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)