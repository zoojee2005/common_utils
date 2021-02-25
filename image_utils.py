import cv2
import IPython
import torch
import numpy as np
import random

from pathlib import Path

# we wrap a new imread function imread
# + Check if passed arg is a Path object, if it is, convert it to string first
# + Make a BGR2RGB transformation automatically
def imread(path, flag = cv2.IMREAD_COLOR):
    # if path is a Path object, it must be converted into string first
    # cv2.imread() cannot accept a Path 
    if isinstance(path, Path):
        path = str(path)
    img = cv2.imread(path, flag)
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

# if input image is in range 0..1, please first multiply img by 255
# assume image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
def IPython_imshow(image, is_RGB2BGR = True, max_size = 640):    
    if isinstance(image, torch.Tensor):
        image = image.cpu().float().numpy()
        
    # un-normalise
    if np.max(image) <= 1:
        image *= 255

    h, w, ch = image.shape
    #resize 
    r = min(float(max_size) / max(h, w), 1.0)  # ratio to limit image size
    image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    #check if RGB=>BGR
    if (ch > 1): #only 3 or 4 channels image can swap RBG channel
        if is_RGB2BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ret = cv2.imencode('.jpg', image) 
    i = IPython.display.Image(data = ret)
    IPython.display.display(i)

#BGR => RGB
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#equivalent codes are:
#     image = image[:, :, ::-1]
#     [H, W, C], the data at last dimension are swapped via ::-1

# assume image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
def im_norm_crop(img, power = 5):
    org_h, org_w = img.shape[0:2]   
    
    h_trim = org_h % 2**power
    w_trim = org_w % 2**power
    
    h_trim = [h_trim // 2, h_trim - h_trim // 2]
    w_trim = [w_trim // 2, w_trim - w_trim // 2]
    
    img = img[h_trim[0]:(org_h - h_trim[1]), w_trim[0]:(org_w - w_trim[1]), ...]
    
    return img

#input image & mask's shape should be [H, W, C]
def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    _, _, channel = mask.shape
    if channel == 1:
        mask = np.dstack((mask, mask, mask))         
    assert channel == 1 or channel == 3, '[error] Input mask\'s channel is {}, only 1 or 3 channels are supported.'.format(channel)    
    
    #set true label to specified color
    mask = mask & color
    
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


#input image & mask's shape should be [H, W, C]
def heatmap_overlay(image, heatmap, heatmap_weight = 0.5, threshold = 0.1): 
    """ 
    Image are seperated to two parts:
    (1) correspond pixels whose value on the heatmap > threshold 
    (2) other pixels on the heatmap <= threshold
    Pixels on original image belong to (2) are keeped and pxiels belong to (1) are replaced by adding weighted image.
    
    # Arguments
        image:      image object (layout is HWC)
        heatmap:    heatmap (layout is HWC)
        heatmap_weight: weight of heatmap to be added into image
        threshold:  pixels whose value are less than threshold in heatmap will be ignored
        
    # Return
        merged image
    """   
    assert image.shape[:2] == heatmap.shape[:2], \
            '[error] image size {} should be equal to heatmap size {}'.format(image.shape, heatmap.shape)
    _, _, channel = heatmap.shape
    if channel == 1:
        heatmap = np.dstack((heatmap, heatmap, heatmap))         
    assert channel == 1 or channel == 3, \
            '[error] Input mask\'s channel is {}, only 1 or 3 channels are supported.'.format(channel)    
    
    #merge image with specified weights    
    if np.max(heatmap) <= 1:
        heatmap *= 255
    heatmap = heatmap.astype(np.uint8)    
    weighted_sum = cv2.addWeighted(heatmap, heatmap_weight, image, (1 - heatmap_weight), 0.)
    
    img = image.copy()
    # find all pixels belong to (1) on original image and replace them with addweighted image
    # if value on any channel is true, replace them on all channels with merged image
    ind = (heatmap > threshold * 255).any(axis = 2)       
    img[ind] = weighted_sum[ind]    
    return img    

# copied from yolo v5
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
