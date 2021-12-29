import numpy as np
from dataset.utils import pose_process
from dataset.dataset import genericDataset as Dataset
import torch

'''
Segmentation Dataset:
    images: n*c*H*W
    annotations: n*num_polygon
'''

class Pose(Dataset):
    def __init__(self, image_url, annotation_url, args) -> None:
        super().__init__(image_url, annotation_url, args)
        self.annotations = np.array(self.annotations['segmentation'])[:, [0, 2, 1]]
        self.labels = self.annotation_to_label(self.annotations)
        self.sigma = 3
        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.thresh = 10

    
    def annotation_to_label(self, annotations):
        """
        annotations: n*2000*3
        output:
            labels: n*1*H*W
        """
        zeros = np.zeros((self.images[0].shape[1:]))
        labels = []
        for annotation in annotations:        
            label = point_process.getcellmap(annotation, zeros)[None, :, :]
            labels.append(label)
        return np.array(labels)

    def label_to_annotation(self, labels):
        '''
        input:
            label: array with shape n*1*H*W
        output:
            annotation: n*2000*3
        '''
        annotations = []
        for label in labels:
            annotation = np.zeros((2000, 3))
            ij = point_process.get_centers(label)
            annotation[:len(ij), 1:] = ij
            annotation[:len(ij), 0] = label[annotation]
            annotations.append(annotation)
        return np.array(annotations)

    def update_annotation(self, prediction):
        '''
        input:
            prediction: array with shape n*1*H*W 
        '''
        self.annotations = self.label_to_annotation(prediction)
    
    def metric(self, prediction):
        '''
        input:
            prediction: array with shape n*1*H*W
        output：

        '''
        pred = self.label_to_annotation(prediction)
        return point_process.metric(pred, self.annotations, self.thresh)


    def __getitem__(self, index):
        batch = super().__getitem__(index)
        image, label = point_process.crop(self.input_size, batch['image'], batch['label'])
        
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     

