import module.loss_fn
import module.unet
import torch
import torch.nn as nn
from torch import optim
import wandb

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        self.backbone = unet.UNet(args.num_channels, args.num_classes)
        if args.data_mode =='point':
            self.criterion = loss_fn.PointLoss(args)
        elif args.data_mode =='pose':
            self.criterion = loss_fn.PoseLoss(args)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=args.lr)
        
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
    def eval(self, dataset, epoch, args):
        self.backbone.eval()
        avg_loss = 0
        output = []
        for batch in dataset:
            imgs = batch['image'].to(device=torch.device('cuda'))
            gt = batch['label'].to(device=torch.device('cuda'))

            pred = self.backbone(imgs)
            loss = self.criterion(pred, gt)
            avg_loss += loss.item() / len(dataset)
            output.append(self.criterion.eval(pred))
        
        metric = dataset.dataset.metric(output)
        wandb.log({'eval loss': avg_loss.item(), "epoch":epoch})
        return metric

