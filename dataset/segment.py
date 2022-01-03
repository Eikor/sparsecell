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
        print('build pose dataset')
        self.annotations = self.annotations['segmentation']
        # self.labels = self.annotation_to_label(self.annotations)
        self.sigma = 3
        self.num_classes = args.num_classes
        self.cellprob_thresh = 0.5
        self.flow_threshold = 0.5
        print('Done.')

    def annotation_to_label(self, annotations):
        """
        input:
            annotations: array with shape n*2*H*W
            [
                image_1: [cell_id, weight]
                image_2: ...
                ... 
            ]
        output:
            label: array with shape n*4*H*W
                [
                    image_1: [weights, cell_prob, dy, dx]
                    image_2: ...
                    ...
                ]
        """
        labels = []
        # build mask
        for annotation in annotations:
            mask, weight = annotation[0], annotation[1]
            flow, _ = pose_process.masks_to_flows(mask)
            label = np.stack([weight, flow], axis=0)
            labels.append(label)
        
        return np.array(labels)

    def label_to_annotation(self, predictions):
        '''
        input:
            predictions: array with shape n*3*H*W
                [
                    image_1: [cell_prob, dy, dx]
                    image_2: ...
                    ...
                ]
        output:
            annotations: n*2*H*W
            [
                image_1: [cell_id, weight]
                image_2: ...
                ... 
            ]
        '''
        annotations = []
        for prediction in predictions:
            cell_prob = prediction[0].sigmoid().cpu().numpy()
            dP = prediction[1:].cpu().numpy()
            p = pose_process.follow_flows(-dP * (cell_prob)/5., niter=200)
            maski = pose_process.get_masks(p, iscell=(cell_prob > self.cellprob_thresh),flows=dP, threshold=self.flow_threshold)
            maski = pose_process.fill_holes_and_remove_small_masks(maski)
            annotations.append(np.stack([maski, cell_prob]))
        return np.array(annotations)

    def update_annotation(self, prediction):
        '''
        input:
            prediction: array with shape n*3*H*W 
        '''
        self.annotations = self.label_to_annotation(prediction)
    
    def metric(self, prediction):
        '''
        input:
            prediction: array with shape n*3*H*W
        output:
            stats: [precision, recall, mean error]
            masks : n*H*W
            [
                mask1, 
                mask2, 
                ...
            ]
        '''
        masks = self.label_to_annotation(prediction)[:, 0]
        gt_masks = self.annotations[:, 0]
        stats = pose_process.metric(masks, gt_masks, self.thresh)

        return stats, masks


    def __getitem__(self, index):
        image = self.images[index]
        anno = self.annotations[index]
        label = self.annotation_to_label([anno])
        image, label = self.crop(self.CROP_SIZE, image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     

