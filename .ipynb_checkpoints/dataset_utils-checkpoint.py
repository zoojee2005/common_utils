import time 
import json

# for project TWTAS, a simpled JSON file does like this
'''
{'image': {'flags': {'sideWall': 'false',
   'plank': 'true',
   'log': 'false',
   'other': 'false'},
  'shapes': [{'label': '3',
    'points': [[417.56955717106274, 601.3209087573092],
     [1168.588941414362, 655.0731804844944]],
    'shape_type': 'rectangle'},
   {'label': '2',
    'points': [[2602.4323454635187, 560.9862089846599],
     [2663.580396274119, 1481.86402439172]],
    'shape_type': 'rectangle'},   
   {'label': '4',
    'points': [[347.92924090666264, 532.1190566118364],
     [3380.2864979398746, 1675.0831660507486]],
    'shape_type': 'rectangle'},
   {'label': '3',
    'points': [[2741.088824497881, 1623.8628521116216],
     [3217.524742892112, 1659.4853020512082]],
    'shape_type': 'rectangle'}],
  'imagePath': '202006150245057_LCam2.jpg',
  'imageHeight': 2048,
  'imageWidth': 4400}}
'''

class PrometheusData:
    def __init__(self, annotation_file = None):                
        if not annotation_file == None:
            tic = time.time()
            print('[Info] Load annotation file into memory.')
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == list, 'annotation file format {} not supported'.format(type(dataset))
            print('[Info] Done (t={:0.2f}s)'.format(time.time() - tic))

            self.dataset = dataset
    
    def getImgInfo(self, index):
        img = self.dataset[index]['image'] 
        path, h, w = img['imagePath'], int(img['imageHeight']), int(img['imageWidth'])                
        return path, h, w
    
    def getAnns(self, index):
        return self.dataset[index]['image']['shapes']            