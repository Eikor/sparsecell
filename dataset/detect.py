import numpy as np
from utils import point_process
from dataset.dataset import genericDataset as Dataset
import torch

class point(Dataset):
    def __init__(self, image_url, annotation_url, args) -> None:
        super().__init__()
        self.images = np.load(image_url)
        self.annotations = np.load(annotation_url)
        self.labels = self.annotation_to_label(self.annotations)
        self.sigma = 3
        self.input_size = args.input_size
        self.num_classes = args.num_classes

    
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
            annotation = point_process.get_centers(label)
            annotations.append(annotation)
        return np.array(annotations)

    def update_annotation(self, prediction):
        '''
        input:
            prediction: array with shape n*1*H*W 
        '''
        self.annotations = self.label_to_annotation(prediction)


    def __getitem__(self, index):
        batch = super().__getitem__(index)
        image, label = point_process.crop(self.input_size, batch['image'], batch['label'])
        
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     
