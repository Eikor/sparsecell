from torch.utils.data import Dataset
import torch
import numpy as np
import json

class genericDataset(Dataset):
    def __init__(self, image_url, annotation_url, args) -> None:
        super().__init__()
        self.CROP_SIZE = args.crop_size
        print('load images')
        self.images = np.load(image_url)
        print('Done.')
        # detection annotation: wights, i, j, h, w
        # segmentation annotation: [x1, y1, x2, y2, ...] 
        f = open(annotation_url, 'r')
        print('load annotations')
        self.annotations = json.load(f)
        print('Done.')
    
    def __len__(self):
        return len(self.images) 

    def annotation_to_label(self, annotation):
        label = annotation
        return label

    def label_to_annotation(self, label):
        annotation = label
        return annotation

    def update_annotation(self, prediction):
        new_annotation = self.label_to_annotation(prediction)
        self.annotations = new_annotation

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

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image, label = self.crop(image, label, self.CROP_SIZE)
        return {
            'image': image, # C*H*W
            'label': label # C*H*W
        }     