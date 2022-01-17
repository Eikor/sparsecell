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

class CoMasking(nn.Module):
    def __init__(self, args):
        super(CoMasking, self).__init__()
        print('build comasking')
        self.wf = unet.UNet(args.num_channels, args.num_classes)
        self.wg = unet.UNet(args.num_channels, args.num_classes)
        self.mask_loss = loss_fn.MaskLoss(args)
        self.flow_loss = loss_fn.FlowLoss(args)
        self.consistence = loss_fn.ConsistLoss(args)
        self.consistence_rate = args.consist
        self.criterion = loss_fn.PoseLoss(args)
        self.optimizer_f = optim.Adam(self.wf.parameters(), lr=args.lr)
        self.optimizer_g = optim.Adam(self.wg.parameters(), lr=args.lr)
        print('Done.')

    def train_step(self, batch, neg_ratio, pos_ratio):
        imgs = batch['image'].to(device=torch.device('cuda'))
        y = batch['label'].to(device=torch.device('cuda'))
        mask = y[:, 0, :, :] == 1
        batch_size, num_channels, H, W = y.shape
        num_neg = H*W*neg_ratio
        num_pos = H*W*pos_ratio

        out_f = self.wf(imgs)
        out_g = self.wg(imgs)
        selected_mask_f = torch.zeros_like(y[:, 0, :, :]) - 1 # -1 represent unselected pixels
        selected_mask_f[mask] = 1
        selected_mask_g = torch.zeros_like(y[:, 0, :, :]) - 1
        selected_mask_g[mask] = 1

        with torch.no_grad():
            # calculate negative loss
            s_loss_f = self.mask_loss(out_f, mask, reduce=None).detach()
            s_loss_g = self.mask_loss(out_g, mask, reduce=None).detach()
            # make sure the ground truth label won't be selected
            s_loss_f[mask] = 999
            s_loss_g[mask] = 999  
            # selection accroding to losses w.r.t negative label
            selected_neg_mask_f = torch.sort(s_loss_f.reshape(batch_size, -1))[0][:, int(num_neg)][:, None, None]
            selected_neg_mask_g = torch.sort(s_loss_g.reshape(batch_size, -1))[0][:, int(num_neg)][:, None, None]
            selected_mask_f[(s_loss_f-selected_neg_mask_f) < 0] = 0
            selected_mask_g[(s_loss_g-selected_neg_mask_g) < 0] = 0

            # calculate positive loss
            s_loss_f[mask==1] = 0
            s_loss_g[mask==1] = 0 
            selected_pos_mask_f = torch.sort(s_loss_f.reshape(batch_size, -1),descending=True)[0][:, int(num_pos)][:, None, None]
            selected_pos_mask_g = torch.sort(s_loss_g.reshape(batch_size, -1),descending=True)[0][:, int(num_pos)][:, None, None]
            
            selected_mask_f[(s_loss_f-selected_pos_mask_f) > 0] = 1
            selected_mask_g[(s_loss_g-selected_pos_mask_g) > 0] = 1

        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        
        consistence = self.consistence(out_f, out_g) * self.consistence_rate
        # consistence.backward(retain_graph=True)

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

        return {
            'loss_f': [mask_loss_f.item(), flow_loss_f.item()], 
            'loss_g': [mask_loss_g.item(), flow_loss_g.item()], 
            'consistence': consistence.item()
            }

    def train_epoch(self, dataset, epoch, args):
        
        self.wf.train()
        self.wg.train()
        avg_loss = 0
        avg_f_mask = 0
        avg_g_mask = 0
        avg_f_flow = 0
        avg_g_flow = 0
        neg_ratio = args.neg_ratio
        pos_ratio = args.pos_ratio
        dataset = tqdm(dataset, desc=f'Epoch: {epoch+1}')

        for batch in dataset:
            losses = self.train_step(batch, neg_ratio, pos_ratio)
            loss_f, loss_g = losses['loss_f'], losses['loss_g']
            consistence = losses['consistence']
            avg_loss += (sum(loss_f)+sum(loss_g) + consistence) / len(dataset)
            avg_f_mask += loss_f[0] / len(dataset)
            avg_g_mask += loss_g[0] / len(dataset)
            avg_f_flow += loss_f[1] / len(dataset)
            avg_g_flow += loss_g[1] / len(dataset)

            dataset.set_postfix({
                'loss_f_mask': '{0:1.5f}'.format(loss_f[0]),
                'loss_g_mask': '{0:1.5f}'.format(loss_g[0]),
                'loss_f_flow': '{0:1.5f}'.format(loss_f[1]),
                'loss_g_flow': '{0:1.5f}'.format(loss_g[1]),
                'consistency': '{0:1.5f}'.format(consistence)
                })
        dataset.close()
        wandb.log({
            'loss_f_mask': avg_f_mask[0],
            'loss_g_mask': avg_g_mask[0],
            'loss_f_flow': avg_f_flow[1],
            'loss_g_flow': avg_g_flow[1],
            'consistency': consistence,
            'train loss': avg_loss
            })
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
        self.wf.eval()
        self.wg.eval()
        avg_loss = 0
        outputs = []
        metric = None
        masks = None
        if args.verbose:
            verbose_flag = True

        dataset = tqdm(dataset, desc=f'Eval Epoch: {epoch+1}')
        for batch in dataset:
            imgs = batch['image'].to(device=torch.device('cuda'))
            gt = batch['label'].to(device=torch.device('cuda'))
            pred = (self.wf(imgs) + self.wg(imgs))/2
            loss = self.criterion(pred, gt)
            avg_loss += loss.item() / len(dataset)
            outputs.append(pred.cpu())
            dataset.set_postfix({
                'loss': '{0:1.5f}'.format(loss.item())
                })
            if verbose_flag:
                verbose_img = imgs[0]
                verbose_flag = False
        outputs = torch.cat(outputs)
        
        if not args.mode == 'test':
            if args.verbose:
                mask = dataset.iterable.dataset.label_to_annotation(outputs[0:1])[:, 1].astype(int)
                self.verbose(verbose_img, mask, epoch, args)
            wandb.log({'eval loss': avg_loss, "epoch":epoch})
            if (epoch+1) % args.save_interval == 0:
                torch.save({
                    'fmodel_state_dict': self.wf.state_dict(),
                    'gmodel_state_dict': self.wg.state_dict(),
                    'foptimizer_state_dict': self.optimizer_f.state_dict(),
                    'goptimizer_state_dict': self.optimizer_g.state_dict(),
                    }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))  
                metric, masks = dataset.iterable.dataset.metric(outputs, args, verbose=True)
                metric_mean = np.mean(metric, axis=0)
                metric_nonzero = np.mean(metric[metric[:, 0]>0], axis=0)
                wandb.log({
                    'val precision': metric_mean[0],
                    'val recall': metric_mean[1],
                    'val iou error': metric_nonzero[2]
                    })
        else:
            metric, masks = dataset.iterable.dataset.metric(outputs, args, verbose=True)
        
        dataset.close()
        
        
        return metric, masks

