import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max

def getgaussianmap(mag, sigma):
    canvas = np.zeros((11, 11))
    canvas[5, 5] = mag
    canvas = gaussian_filter(canvas, sigma, mode='constant')
    canvas = canvas / np.max(canvas)
    return canvas


def getcellmap(frame, zeros, gaussian_map):
    '''
    frame: 2000*3 
    '''
    cmap = zeros.copy()
    accumulate_map = zeros.copy()
    b = np.array(zeros.shape[0:2])-1
    r = ((gaussian_map.shape[0]) - 1)/2
    
    cells = frame.astype(int)
    for weights, xcord, ycord in cells:
        if weights > 1:
            weights = 1
        tl = np.array([int(ycord) - r, int(xcord) - r]).astype('int')
        br = np.array([int(ycord) + r, int(xcord) + r]).astype('int')
        shift1 = tl
        shift2 = br
        if any(tl<0):
            shift1 = np.maximum(tl, 0) - tl
            tl = np.maximum(tl, 0)
            # cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map[shift1[0]:, shift1[1]:] * weights
        elif any(br - b > 0):
            shift2 = (2*r+1 + np.minimum(br, b) - br).astype('int')
            br = np.minimum(br, b)
            # cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map[:shift2[0], :shift2[1]] * weights
        else:
            cmap[tl[0]:br[0]+1, tl[1]:br[1]+1] = gaussian_map * weights
        accumulate_map = np.maximum(cmap, accumulate_map)        
    return accumulate_map

def get_centers(label, pred_th):
    '''
    label: 1*H*W
    '''
    th = pred_th
    anno = peak_local_max(label, min_distance=5, threshold_abs=th)[:, ::-1]
    return anno

def crop(crop_size, img, label):
    if crop_size < 0:
        return img, label
    
    h, w = img.shape[1:3]        
    # random crop images
    tl = np.array([np.random.randint(0, h-crop_size),
                np.random.randint(0, w-crop_size)])
    br = tl + crop_size

    ### crop image ###
    img_patch, label_patch = img[:, tl[0]:br[0], tl[1]:br[1]], label[:, tl[0]:br[0], tl[1]:br[1]]
    
    return img_patch, label_patch