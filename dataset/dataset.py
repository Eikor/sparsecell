from torch.utils.data import Dataset
import numpy as np
import os

class genericDataset(Dataset):
    def __init__(self, image_url, annotation_url, args) -> None:
        super().__init__()
        self.CROP_SIZE = args.crop_size
        print('load images')
        self.images = np.load(image_url)
        self.images = self.images / np.max(self.images)
        if self.images.shape[1] > 3:
            print('add channel axis')
            self.images = self.images[:, None, :, :]
        print('Done.')
        # detection annotation: wights, i, j, h, w
        # segmentation annotation: [x1, y1, x2, y2, ...] 
        print('load annotations')
        self.annotations = np.load(annotation_url)
        self.label_url = os.path.join(os.path.split(annotation_url)[0], f'anno_label{args.anno_rate}')
        print('Done.')
    
    def __len__(self):
        return len(self.annotations) 

    def annotation_to_label(self, annotation):
        label = annotation
        return label

    def label_to_annotation(self, label):
        annotation = label
        return annotation

    def update_annotation(self, prediction):
        new_annotation = self.label_to_annotation(prediction)
        self.annotations = new_annotation

    def crop(self, img, label):
        if self.CROP_SIZE < 0:
            return img, label
        if len(label.shape)==4:
            label = label.squeeze()
        crop_size = self.CROP_SIZE
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
        image, label = self.crop(image, label)
        return {
            'image': image, # C*H*W
            'label': label # C*H*W
        }     