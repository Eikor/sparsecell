import numpy as np
from dataset.utils import pose_process
from dataset.dataset import genericDataset as Dataset
import torch
import cv2

'''
Segmentation Dataset:
    images: n*c*H*W
    annotations: n*num_polygon
'''

class Pose(Dataset):
    def __init__(self, image_url, annotation_url, args) -> None:
        super().__init__(image_url, annotation_url, args)
        segmentation = self.annotations['segmentation']
        self.annotations = segmentation
        self.labels = self.annotation_to_label(self.annotations)
        self.sigma = 3
        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.cellprob_thresh = 0.5
        self.flow_threshold = 0.5

    def annotation_to_label(self, annotations):
        """
        input:
            annotations: [
                image_1: [[cell1], [cell2], ...]
                image_2: ...
                ... 
            ]
        output:
            label: array with shape n*3*H*W
                [
                    image_1: [cell_prob, dy, dx]
                    image_2: ...
                    ...
                ]
        """
        labels = []
        # build mask
        masks = []
        mask = np.zeros_like(self.images[0])
        for image in annotations:
            for i in range(len(image)):
                contours = np.around(np.array(image[i])).astype(int).reshape(-1, 2)
                mask = cv2.fillPoly(mask, [contours], i+1)
            masks.append(mask)
        # compute flow
        for mask in masks:
            flow, _ = pose_process.masks_to_flows(mask)
            labels.append(flow)

        return np.array(labels)

    def label_to_annotation(self, labels):
        '''
        input:
            label: array with shape n*3*H*W
                [
                    image_1: [cell_prob, dy, dx]
                    image_2: ...
                    ...
                ]
        output:
            annotations: [
                image_1: [[cell1], [cell2], ...]
                image_2: ...
                ... 
            ]
        '''
        annotations = []
        for label in labels:
            annotation = []
            cell_prob = label[0] > self.cellprob_thresh
            dP = label[1:]
            p = pose_process.follow_flows(-dP * (cell_prob)/5., niter=200)
            maski = pose_process.get_masks(p, iscell=(cell_prob),flows=dP, threshold=self.flow_threshold)
            maski = pose_process.fill_holes_and_remove_small_masks(maski)

            annotations.append(annotation)
        return annotations

    def update_annotation(self, prediction):
        '''
        input:
            prediction: array with shape n*3*H*W 
        '''
        self.annotations = self.label_to_annotation(prediction)
    
    def metric(self, prediction):
        '''
        input:
            prediction: array with shape n*1*H*W
        output：

        '''
        pred = self.label_to_annotation(prediction)
        return pose_process.metric(pred, self.annotations, self.thresh)


    def __getitem__(self, index):
        batch = super().__getitem__(index)
        return {
            'image': torch.FloatTensor(batch['image']),
            'label': torch.FloatTensor(batch['label'])
        }     

