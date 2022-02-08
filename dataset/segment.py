import numpy as np
from dataset.utils import pose_process
from dataset.dataset import genericDataset as Dataset
import torch
import cv2
from tqdm import tqdm
import os

class Pose(Dataset):
    def __init__(self, image_url, annotation_url, label_url, args, Aug=True) -> None:
        super().__init__(image_url, annotation_url, label_url, args)
        print('build pose dataset')
        self.annotations = self.annotations['segmentation']

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
        self.flow_threshold = 1
        self.iou_thresh = 0.5
        self.pose_alpha = args.pose_alpha
        self.pose_beta = args.pose_beta
        self.non_linear = args.nonlinear_flow
        self.c = args.flow_c
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
            cell_prob = prediction[0].sigmoid().numpy()
            if self.non_linear:
                dP = torch.tanh(self.c * prediction[1:]).cpu().numpy()
            else:
                dP = prediction[1:].cpu().numpy()
            if self.pose_alpha == 0:
                p = pose_process.follow_flows(-dP, niter=200)
                maski = pose_process.get_masks(p, iscell=None, flows=None, threshold=self.flow_threshold)
            else:
                p = pose_process.follow_flows(-dP * (cell_prob), niter=200)
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
    
    def metric(self, prediction, args, verbose=False):
        '''
        metric top 100 prediction
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
        if args.mode != 'test':
            masks = self.label_to_annotation(prediction[0:100])[:, 1].astype(int)
        else:
            masks = self.label_to_annotation(prediction)[:, 1].astype(int)
        gt_masks = self.annotations[:, 1]
        print('calculate iou')
        stats = pose_process.metric(masks, gt_masks, self.iou_thresh)
        if verbose:
            test_url = args.save_dir + '/test'
            try:
                os.makedirs(test_url)
            except:
                pass

            for i, (mask, gt_mask) in enumerate(zip(masks, gt_masks)):
                img = self.images[i]
                img_channels = img.shape[0]
                if img_channels == 1:
                    img = cv2.cvtColor((img.transpose(1, 2, 0)*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
                elif img_channels == 2:
                    zeros = np.zeros_like(img[0:1])
                    img = np.concatenate([img, zeros]).transpose(1, 2, 0)
                canvas = img
                mask_ = img
                mask_[mask>0, 2] = 255
                mask_[gt_mask>0, 1] = 255
                canvas = cv2.addWeighted(canvas, 0.8, mask_, 0.2, 0)
                cv2.imwrite(test_url + f'/{i}.jpg', canvas)
        stats = np.array(stats)
        return stats, masks


    def __getitem__(self, index):
        image = self.images[index]
        label_url = os.path.join(self.label_url, f'{index}.npy')
        if os.path.exists(label_url):
            try:
                label = np.load(label_url)
            except:
                print(index)
                anno = self.annotations[index]
                label = self.annotation_to_label([anno]).squeeze()
            if len(label.shape) == 4:
                label = label[0]
        else:
            anno = self.annotations[index]
            label = self.annotation_to_label([anno]).squeeze()
        image, label = self.crop(image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     

class SoftPose(Dataset):
    def __init__(self, image_url, annotation_url, label_url, args, Aug=True) -> None:
        super(SoftPose, self).__init__(image_url, annotation_url, label_url, args)
        print('build softpose dataset')
        self.annotations = self.annotations['segmentation']
        self.sigma = 1.5
        self.kernel_size = (5, 5)
        self.num_classes = args.num_classes
        self.cellprob_thresh = 0.5
        self.flow_threshold = 1
        self.iou_thresh = 0.5
        self.pose_alpha = args.pose_alpha
        self.pose_beta = args.pose_beta
        self.non_linear = args.nonlinear_flow
        self.c = args.flow_c

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
            flow, flow_cb = pose_process.masks_to_flows(mask)
            center, boundary = flow_cb
            center = cv2.GaussianBlur(center, self.kernel_size, self.sigma)
            scale = np.max(center)
            center = center / scale
            boundary = boundary / scale / 3
            label = np.concatenate([center[None, :, :], boundary[None, :, :], flow], axis=0)
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
            cell_prob = prediction[0].sigmoid().numpy()
            if self.non_linear:
                dP = torch.tanh(self.c * prediction[1:]).cpu().numpy()
            else:
                dP = prediction[1:].cpu().numpy()
            if self.pose_alpha == 0:
                p = pose_process.follow_flows(-dP, niter=200)
                maski = pose_process.get_masks(p, iscell=None, flows=None, threshold=self.flow_threshold)
            else:
                p = pose_process.follow_flows(-dP * (cell_prob), niter=200)
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
    
    def metric(self, prediction, args, verbose=False):
        '''
        metric top 100 prediction
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
        if args.mode != 'test':
            masks = self.label_to_annotation(prediction[0:100])[:, 1].astype(int)
        else:
            masks = self.label_to_annotation(prediction)[:, 1].astype(int)
        gt_masks = self.annotations[:, 1]
        print('calculate iou')
        stats = pose_process.metric(masks, gt_masks, self.iou_thresh)
        if verbose:
            test_url = args.save_dir + '/test'
            try:
                os.makedirs(test_url)
            except:
                pass

            for i, (mask, gt_mask) in enumerate(zip(masks, gt_masks)):
                img = self.images[i]
                img_channels = img.shape[0]
                if img_channels == 1:
                    img = cv2.cvtColor((img.transpose(1, 2, 0)*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
                elif img_channels == 2:
                    zeros = np.zeros_like(img[0:1])
                    img = np.concatenate([img, zeros]).transpose(1, 2, 0)
                canvas = img
                mask_ = img
                mask_[mask>0, 2] = 255
                mask_[gt_mask>0, 1] = 255
                canvas = cv2.addWeighted(canvas, 0.8, mask_, 0.2, 0)
                cv2.imwrite(test_url + f'/{i}.jpg', canvas)
        stats = np.array(stats)
        return stats, masks


    def __getitem__(self, index):
        image = self.images[index]
        label_url = os.path.join(self.label_url, f'{index}.npy')
        if os.path.exists(label_url):
            try:
                label = np.load(label_url)
            except:
                print(index)
                anno = self.annotations[index]
                label = self.annotation_to_label([anno]).squeeze()
            if len(label.shape) == 4:
                label = label[0]
        else:
            anno = self.annotations[index]
            label = self.annotation_to_label([anno]).squeeze()
        image, label = self.crop(image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     