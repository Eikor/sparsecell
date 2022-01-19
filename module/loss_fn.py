from turtle import pos
import torch
import torch.nn as nn
from dataset.utils import point_process

class PointLoss(nn.Module):
    def __init__(self, args):
        super(PointLoss, self).__init__()

    def forward(self, y_hat, y):
        prob = torch.sigmoid(y_hat).clamp(min=1e-4, max=(1 - 1e-4))
        loss = torch.mean(torch.log(prob) * (y==1) + torch.log(1 - prob) * (1-y))
        return -loss

    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat).clamp(min=1e-4, max=(1 - 1e-4))
    #     return {'output': prob.detach().cpu().numpy()}          

class PoseLoss(nn.Module):
    def __init__(self, args):
        super(PoseLoss, self).__init__()
        self.alpha = args.pose_alpha
        self.beta = args.pose_beta
        self.thresh = 0.5
        if args.pose_loss == 'l1':
            self.l2 = nn.L1Loss(reduction='none')
        else:
            self.l2 = nn.MSELoss(reduction='none')
   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
        weights: 
            1 for cell；
            (0, 1) for low-confidence cell;
            0 for unlabeled pixel, 50-50
            [-1, 0) for background
        '''
        prob = torch.sigmoid(y_hat[:, 0:1, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        flow = y_hat[:, 1:, :, :]
        weights = y[:, 0:1, :, :]
        gt_flow = y[:, 1:, :, :]
        
        pos_mask = weights > self.thresh
        if torch.any(weights<0):
            neg_mask = weights < 0
        else: neg_mask = weights == 0
        select_mask = pos_mask + neg_mask

        prob_loss = -torch.sum(torch.log(prob) * pos_mask + torch.log(1 - prob) * (neg_mask)) / torch.sum(select_mask)
    
        if masked:
            if torch.sum(pos_mask) == 0:
                flow_loss = torch.sum(self.l2(flow, gt_flow) * pos_mask)
            else:
                flow_loss = torch.sum(self.l2(flow, gt_flow) * pos_mask) / torch.sum(pos_mask)
        else:
            flow_loss = torch.sum(self.l2(flow, gt_flow) * select_mask) / torch.sum(select_mask)

        return self.beta * flow_loss + self.alpha * prob_loss
    
    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
    #     flow = y_hat[:, 1:, :, :]
    #     return {
    #         'output': prob.detach().cpy().numpy(),
    #         'flow': flow.detach().cpu().numpy()
    #     }

class MaskLoss(nn.Module):
    def __init__(self, args):
        super(MaskLoss, self).__init__()
        self.alpha = args.pose_alpha
        self.beta = args.pose_beta
        self.thresh = 0.5
   
    def forward(self, y_hat, y, reduce='mean'):
        '''
        weights: 
            1 for cell；
            (0, 1) for low-confidence cell;
            0 for unlabeled pixel, 50-50
            [-1, 0) for background
        '''
        pos_mask = y == 1
        neg_mask = y == 0
        prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        prob_loss = torch.log(prob) * pos_mask + torch.log(1 - prob) * neg_mask
        if reduce == 'mean':
            return - torch.sum(prob_loss) / torch.sum(pos_mask + neg_mask)
        else:
            return -prob_loss    

class FlowLoss(nn.Module):
    def __init__(self, args):
        super(FlowLoss, self).__init__()
        self.beta = args.pose_beta
        self.l2 = nn.L1Loss(reduction='none')
   
    def forward(self, y_hat, y, reduce='mean'):
        flow = y_hat[:, 1:, :, :]
        weights = y[:, 0:1, :, :]
        gt_flow = y[:, 1:, :, :]
        
        pos_mask = weights == 1
        loss = self.beta * self.l2(flow, gt_flow) * pos_mask
        if reduce == 'mean':
            if torch.sum(pos_mask) == 0:
                return torch.sum(loss)
            return torch.sum(loss) / torch.sum(pos_mask)
        else:
            return loss

class ConsistLoss(nn.Module):
    def __init__(self, args):
        super(ConsistLoss, self).__init__()
        self.consist = args.consist
        self.l2 = nn.MSELoss(reduce=None)
   
    def forward(self, y_f, y_g, reduce='mean'):
        flow_f = y_f[:, 1:, :, :]
        flow_g = y_g[:, 1:, :, :]

        loss = self.consist * self.l2(flow_f, flow_g)
        if reduce == 'mean':
            return torch.mean(loss)
        else:
            return loss