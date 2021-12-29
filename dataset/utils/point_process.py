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
    input:
        label: 1*H*W
    
    '''
    th = pred_th
    ij = peak_local_max(label, min_distance=5, threshold_abs=th)[:, ::-1]
    return ij

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

def compare_(pred, anno, patience):
    if len(pred) == 0:
        return 0, 0, 0
    anno = anno[anno[:, 0]>0]
    pred = pred[pred[:, 0]>0]
    error = anno[:, np.newaxis, :] - pred[np.newaxis, :, :]
    error = np.linalg.norm(error, axis=-1)
  
    pred_error = np.min(error, axis=0) # find closest cell of each prediction 
    association = np.argmin(error, axis=0) 
    association_error = np.min(error, axis=1)
    association_error = association_error[association_error <= patience]
    # each prediction only has one valid associatin target 
    TP = min(np.sum(pred_error <= patience), len(np.unique(association)))
    
    precision = TP / len(pred)
    recall = TP / len(anno)
    if len(association_error) == 0:
        return 0, 0, 0
    return precision, recall, np.mean(association_error)


def metric(preds, annos, thresh):
    '''
    input:
        preds: numpy array with shape n*2000*3
        annos: numpy array with shape n*2000*3
    '''
    assert len(preds) == len(annos)
    stat = []
    for pred, anno in zip(preds, annos): # for each image
        precision, recall, error = compare_(pred, anno, thresh)
        stat.append(np.array([precision, recall, error]))
    return stat
