import cv2
import IPython
import torch
import numpy as np

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