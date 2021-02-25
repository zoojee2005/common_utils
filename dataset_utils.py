import time 
import json
import os
import torch
from pathlib import Path

from .image_utils import imread

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

class PrometheusData(torch.utils.data.Dataset):
    def __init__(self, annotation_file = None):                
        if not annotation_file == None:
            tic = time.time()
            print('[Info] Load annotation file into memory.')
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == list, 'annotation file format {} not supported'.format(type(dataset))
            print('[Info] Done (t={:0.2f}s)'.format(time.time() - tic))

            self.dataset = dataset    
    
    def __getitem__(self, index):
        img = self.dataset[index]['image'] 
        path, h, w = img['imagePath'], int(img['imageHeight']), int(img['imageWidth'])                
        shapes = self.dataset[index]['image']['shapes']
        return (path, h, w), shapes     
    
    def __len__(self):
        return len(self.dataset)

'''
Customized annotations from Visi tech.
All annotation information is put in *.text file. 
The *.txt file specifications are:
+ One row per image
+ Each row is `image class x_lefttop y_lefttop width height` format.
+ All objects are place in one row
+ Box coordinates are in pixels
    
    the following text is a simple example.
    
    201103-150320-000226_1.jpg 4 171 178 415 382 4 413 146 824 224 4 211 329 642 416 4 509 489 806 546
    
    There are 4 objects in image 201103-150320-0000226_1.jpg.
'''    
class VisiData(torch.utils.data.Dataset):
    def __init__(self, annotation_file):
        with open(annotation_file) as f:
            self.anns = f.readlines()
            
    def __getitem__(self, index):
        # split msg with space char
        metas = self.anns[index].split()   
        tuple_len = 5  # (class_id, x, y, w, h)
        assert (len(metas) - 1) % tuple_len == 0, \
        'Error, annotation msg\'s length {} are not divisible by 5'.format(len(metas) - 1)  
        image = metas[0]   # the first one is image
        
        # number of annos in this image
        num_annos = (len(metas) - 1) // tuple_len        
        annos = []
        
        start_index = 1        
        for index in range(0, num_annos):            
            # append [class_id, x, y, w, h] 
            annos.append(metas[start_index : start_index + tuple_len])         
            start_index += tuple_len            
        
        return image, annos
    
    def __len__(self):
        return len(self.anns)    
    
'''
## convert to YOLO format
After using a tool like CVAT, makesense.ai or Labelbox to label your images, 
export your labels to YOLO format, with one *.txt file per image (if no objects in image, no *.txt file is required). 
The *.txt file specifications are:
+ One row per object
+ Each row is `class x_center y_center width height` format.
+ Box coordinates must be in **normalized xywh format** (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
+ Class numbers are zero-indexed (start from 0).
'''
def PromAnno2Yolo(yolo_anno_path, prometheus_dataset):
    # make sure that yolo_anno_path is a Path object
    yolo_anno_path = Path(yolo_anno_path) if isinstance(yolo_anno_path, str) else yolo_anno_path
    # check if path exists
    if not os.path.exists(yolo_anno_path):
        os.mkdir(yolo_anno_path)    

    # create label files for all images
    yolo_files = []
    for i, ((img_path, img_h, img_w), anns) in enumerate(prometheus_dataset):         
        assert img_w > 0 and img_h > 0, 'images\'s width = {} and height = {} should be greater than zero.'.format(img_w, img_h)
        #generate yolo's label file
        label_file = (yolo_anno_path / img_path).with_suffix('.txt')
        with open(label_file, 'w', encoding = 'utf-8') as file:
            for ann in anns:
                cls_id = ann['label']
                assert ann['shape_type'] == 'rectangle', 'label\'s shape type {} should be rectangle.'.format(ann['shape_type'])                
                # flatten ann['points'] list, (x1, y1), (x2, y2) = ann['points'][0], ann['points'][1]         
                x = [x for point in ann['points'] for x in point]

                # from (x1, y1, x2, y2) => (x_center, y_center, width, height)
                x = [float(x) for x in yolo.xyxy2xcycwh(x)]
                # normalize (x1, y1) and (w, h)                    
                y = yolo.normalize(x, img_w, img_h)
                # write to file
                file.write('{} {} {} {} {} \n'.format(cls_id, *y))
        # create yolo file list
        yolo_files.append(label_file)
    
    #
    return yolo_files

class yolo():
    @staticmethod
    def xyxy2xcycwh(x):
        #from (x1, y1, x2, y2) => (x_center, y_center, width, height)   
        y = x.copy()
        w, h = (x[2] - x[0]), (x[3] - x[1])
        y = [(x[0] + w / 2), (x[1] + h / 2), w, h]
        assert w > 0 and h > 0, 'w = {}, h = {} must be greater than zero'.format(w, h)    
        return y
    
    @staticmethod
    def xcycwh2xyxy(x):
        #from (x_center, y_center, width, height) =>  (x1, y1, x2, y2)
        y = x.copy()
        w, h = x[2], x[3]
        y = [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2]
        assert w > 0 and h > 0, 'w = {}, h = {} must be greater than zero'.format(w, h)    
        return y
    
    @staticmethod
    def normalize(x, img_w, img_h):
        #normalize (x1, y1) and (w, h)                    
        #x_c, y_c, w, h = x_c / img_w, y_c / img_h, w / img_w, h / img_h
        y = x.copy()
        y = [x[0] / img_w, x[1] / img_h, x[2] / img_w, x[3] / img_h]
        return y
        
    @staticmethod
    def unnormalize(x, img_w, img_h):
        y = x.copy()
        y = [x[0] * img_w, x[1] * img_h, x[2] * img_w, x[3] * img_h]    
        return y
                
        
# similar to function: PromAnno2Yolo    
def VisiAnno2Yolo(yolo_anno_path, image_path, visi_dataset, start = 0, length = 0):
    """ 
    A dataset constructed using visi's annotation file:
    
    # Arguments
        yolo_anno_path:      where generated yolo anno files are put
        image_path:          where images of datasets are stored
        visi_dataset:        A dataset constructed using visi's annotation file
        start = 0:           Cusor points to where to start in dataset
        len = 0:             Number of items to be converted. if 0, all items will be processed.
        
    # Return
        merged image
    """   
    #check if path exists
    if not os.path.exists(yolo_anno_path):
        os.mkdir(yolo_anno_path)    
    
    # create label files for all images
    yolo_files = []
    # check if start and end index are valid
    end = len(visi_dataset) if length == 0 or (start + length) > len(visi_dataset) else (start + length)        
    for index in range(start, end):         
        (image, anns) = visi_dataset[index]
        img = imread(image_path / image)
        img_h, img_w, _ = img.shape
        
        #generate yolo's label file
        label_file = (yolo_anno_path / image).with_suffix('.txt')        
        with open(label_file, 'w', encoding = 'utf-8') as file:
            for ann in anns:
                #cls_id, x1, y1, x2, y2 = [int(x) for x in ann]                                
                x = [int(x) for x in ann]                                

                #from (x1, y1, x2, y2) => (x_center, y_center, width, height)                
                x = [float(x) for x in yolo.xyxy2xcycwh(x[1:])]
                
                #normalize (x1, y1) and (w, h)                    
                y = yolo.normalize(x, img_w = img_w, img_h = img_h)                
                
                #write to file
                file.write('{} {} {} {} {} \n'.format(x[0] - 1, *y))   # class id start from 0, but visi's data do start from 1
        # create yolo file list
        yolo_files.append(label_file)
    
    #
    return yolo_files