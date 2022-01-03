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

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        print('build networks')
        self.backbone = unet.UNet(args.num_channels, args.num_classes)
        if args.data_mode =='point':
            self.criterion = loss_fn.PointLoss(args)
        elif args.data_mode =='pose':
            self.criterion = loss_fn.PoseLoss(args)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=args.lr)
        print('Done.')
        
    def train(self, dataset, epoch, args):
        self.backbone.train()
        avg_loss = 0
        for batch in dataset:
            imgs = batch['image'].to(device=torch.device('cuda'))
            gt = batch['label'].to(device=torch.device('cuda'))

            loss = self.criterion(self.backbone(imgs), gt)
            self.optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item() / len(dataset)
            self.optimizer.step()

        wandb.log({'train loss': avg_loss.item(), "epoch":epoch})
    
    @torch.no_grad()
    def verbose(self, img, mask, epoch, args):
        '''
        Input:
            img: C*H*W tensor
            pred: C*H*W tensor
        Output:
            None
        '''
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
            cv2.imwrite('input.jpg', verbose_img)
        
        # save mask
        output = cv2.addWeighted(verbose_img, 0.8, mask, 0.2)
        cv2.imwrite(f'epoch_{epoch}.jpg', output)

    @torch.no_grad()
    def eval(self, dataset, epoch, args):
        self.backbone.eval()
        avg_loss = 0
        outputs = []
        if epoch % args.save_interval == 0:
            torch.save({
                'model_state_dict': self.backbone.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))  
        if args.verbose:
            verbose_flag = True
        for batch in dataset:
            imgs = batch['image'].to(device=torch.device('cuda'))
            gt = batch['label'].to(device=torch.device('cuda'))
            pred = self.backbone(imgs)
            loss = self.criterion(pred, gt)
            avg_loss += loss.item() / len(dataset)
            outputs.append(pred)
            if verbose_flag:
                verbose_img = imgs[0]
                verbose_flag = False

        wandb.log({'eval loss': avg_loss.item(), "epoch":epoch})
        outputs = torch.cat(outputs)
        metric, masks = dataset.dataset.metric(outputs)
        if args.verbose:
            self.verbose(verbose_img, masks[0])
        return metric

