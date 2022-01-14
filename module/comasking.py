from numpy.lib.shape_base import expand_dims
import module.loss_fn as loss_fn
import module.unet as unet
import torch
import torch.nn as nn
from torch import optim
import wandb
import os
import cv2
import numpy as np
from tqdm import tqdm

class CoMasking():
    def __init__(self, args):
        super(CoMasking, self).__init__()
        print('build comasking')
        self.wf = unet.UNet(args.num_channels, args.num_classes)
        self.wg = unet.UNet(args.num_channels, args.num_classes)
        self.mask_loss = None
        self.flow_loss = None
        self.optimizer_f = optim.Adam(self.wf.parameters(), lr=args.lr)
        self.optimizer_g = optim.Adam(self.wg.parameters(), lr=args.lr)
        print('Done.')

    def train_step(self, batch, ratio):
        imgs = batch['image'].to(device=torch.device('cuda'))
        y = batch['label'].to(device=torch.device('cuda'))
        mask = y[:, 0, :, :] == 1
        batch_size, num_channels, H, W = y.shape
        num_selected = H*W*ratio
        out_f = self.wf(imgs)
        out_g = self.wg(imgs)

        with torch.no_grad():
            # calculate negative loss
            neg_loss_f = self.mask_loss(out_f, mask).detach()
            neg_loss_g = self.mask_loss(out_g, mask).detach()
            # make sure the ground truth label won't be selected
            neg_loss_f[mask==1] = 999
            neg_loss_g[mask==1] = 999  
            # selection accroding to losses w.r.t negative label
            selected_neg_mask_f = torch.topk(neg_loss_f, int(num_selected), largest=False)[1]
            selected_neg_mask_g = torch.topk(neg_loss_g, int(num_selected), largest=False)[1]

            # calculate positive loss
            pos_loss_f = self.mask_loss(self.wf(imgs), 1-mask).detach()
            pos_loss_g = self.mask_loss(self.wg(imgs), 1-mask).detach()
            pos_loss_f[mask==1] = 999
            pos_loss_g[mask==1] = 999 
            selected_pos_mask_f = torch.topk(pos_loss_f, int(num_selected), largest=False)[1]
            selected_pos_mask_g = torch.topk(pos_loss_g, int(num_selected), largest=False)[1]
            
            selected_mask_f = torch.zeros_like(mask)
            selected_mask_f[mask] = 1
            selected_mask_f[selected_pos_mask_f] = 1
            selected_mask_f[selected_neg_mask_f] = 0
            selected_mask_g = torch.zeros_like(mask)
            selected_mask_g[mask] = 1
            selected_mask_g[selected_pos_mask_g] = 1
            selected_mask_g[selected_neg_mask_g] = 0

        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        
        mask_loss_f = self.mask_loss(out_f, selected_mask_g)
        mask_loss_g = self.mask_loss(out_g, selected_mask_f)

        flow_loss_f = self.flow_loss(out_f, y)
        flow_loss_g = self.flow_loss(out_g, y)

        loss_f = mask_loss_f + flow_loss_f
        loss_g = mask_loss_g + flow_loss_g
        loss_f.backward()
        loss_g.backward()
        
        self.optimizer_f.step()
        self.optimizer_g.step()

        return (loss_g.item() + loss_f.item())/2

    def train_epoch(self, dataset, epoch, args):
        
        self.wf.train()
        self.wg.train()
        avg_loss = 0
        ratio = None
        dataset = tqdm(dataset, desc=f'Epoch: {epoch+1}')

        for batch in dataset:
            loss = self.train_step(batch, ratio)
            avg_loss += loss / len(dataset)
            
            dataset.set_postfix({
                'loss': '{0:1.5f}'.format(loss)
                })
        dataset.close()
        wandb.log({'train loss': avg_loss, "epoch":epoch})
        return avg_loss
    
    @torch.no_grad()
    def verbose(self, img, mask, epoch, args):
        '''
        Input:
            img: C*H*W tensor
            mask: H*W int array 
        Output:
            None
        '''
        mask = mask.squeeze()
        verbose_url = args.save_dir + '/verbose'
        verbose_img = (img.cpu().numpy()*255).astype('uint8')
        img_channels = img.shape[0]
        # convert img to rgb
        if img_channels == 1:
            verbose_img = cv2.cvtColor(verbose_img.transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)
        elif img_channels == 2:
            zeros = np.zeros_like(verbose_img[0:1])
            verbose_img = np.concatenate([verbose_img, zeros]).transpose(1, 2, 0)

        # save verbose img
        try:
            os.makedirs(verbose_url)
        except:
            pass
        if not os.path.exists(verbose_url+'/input.jpg'):
            cv2.imwrite(verbose_url+'/input.jpg', verbose_img)
        
        canvas = verbose_img
        canvas[mask>0, 0] = 255
        # save mask
        output = cv2.addWeighted(verbose_img, 0.8, canvas, 0.2, 1)
        cv2.imwrite(verbose_url+f'/epoch_{epoch+1}.jpg', output)

    @torch.no_grad()
    def eval(self, dataset, epoch, args):
        self.backbone.eval()
        avg_loss = 0
        outputs = []
        metric = None
        masks = None

        
        return metric, masks

