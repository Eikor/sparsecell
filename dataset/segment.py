import numpy as np
from dataset.utils import pose_process
from dataset.dataset import genericDataset as Dataset
import torch
import cv2
from tqdm import tqdm
import os

class Pose(Dataset):
    def __init__(self, image_url, annotation_url, args, Aug=True) -> None:
        super().__init__(image_url, annotation_url, args)
        print('build pose dataset')
        self.annotations = self.annotations['segmentation']
        # self.labels = self.annotation_to_label(self.annotations)
        if os.path.exists(self.label_url):
            print('check existing label buffer')
            for i in range(len(self.annotations)):
                if not os.path.exists(self.label_url+f'/{i}.npy'):
                    print(f'build label {i}')
                    label = self.annotation_to_label([self.annotations[i]])
                    np.save(self.label_url + f'/{i}.npy', label)
        else:
            os.makedirs(self.label_url)
            self.annotation_to_label(self.annotations, self.label_url)
        if not Aug:
            self.CROP_SIZE = -1
        self.sigma = 3
        self.num_classes = args.num_classes
        self.cellprob_thresh = 0.5
        self.flow_threshold = 0.5
        self.iou_thresh = 0.5
        print('Done.')

    def annotation_to_label(self, annotations, save_dir=None):
        """
        input:
            annotations: array with shape n*2*H*W
            [
                image_1: [weight, cellid]
                image_2: ...
                ... 
            ]
        output:
            label: array with shape n*4*H*W
                [
                    image_1: [weights, dy, dx]
                    image_2: ...
                    ...
                ]
        """
        labels = []
        # build mask
        for i, annotation in tqdm(enumerate(annotations), desc='Calculate flows', total=len(annotations)):
            weight, mask = annotation[0], annotation[1]
            flow, _ = pose_process.masks_to_flows(mask)
            label = np.concatenate([weight[None, :, :], flow], axis=0)
            labels.append(label)
            if save_dir is not None:
                np.save(save_dir + f'/{i}.npy', label)

        return np.array(labels)

    def label_to_annotation(self, predictions, mode='get'):
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
                image_1: [weight, cell_id]
                image_2: ...
                ... 
            ]
        '''
        annotations = []
        for prediction in tqdm(predictions, desc='Follow flows'):
            cell_prob = prediction[0].sigmoid().cpu().numpy()
            dP = prediction[1:].cpu().numpy()
            p = pose_process.follow_flows(-dP * (cell_prob)/5., niter=200)
            maski = pose_process.get_masks(p, iscell=(cell_prob > self.cellprob_thresh),flows=dP, threshold=self.flow_threshold)
            maski = pose_process.fill_holes_and_remove_small_masks(maski)
            annotations.append(np.stack([cell_prob, maski]))
        return np.array(annotations)

    def update_annotation(self, prediction):
        '''
        input:
            prediction: array with shape n*3*H*W 
        '''
        self.annotations = self.label_to_annotation(prediction)
        self.label_url = os.path.join(os.path.split(self.label_url)[0], 'pseudo_label')
        self.annotation_to_label(self.annotations, self.label_url)
    
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
        masks = self.label_to_annotation(prediction[0:20])[:, 1].astype(int)
        gt_masks = self.annotations[:, 0]
        print('calculate iou')
        stats = pose_process.metric(masks, gt_masks, self.iou_thresh)

        return stats, masks


    def __getitem__(self, index):
        image = self.images[index]
        label_url = os.path.join(self.label_url, f'{index}.npy')
        if os.path.exists(label_url):
            label = np.load(label_url)
        else:
            anno = self.annotations[index]
            label = self.annotation_to_label([anno]).squeeze()
        image, label = self.crop(image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     

