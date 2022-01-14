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
        self.l2 = nn.MSELoss(reduce=None)
   
    def forward(self, y_hat, y, reduction='sum'):
        '''
        weights: 
            1 for cell；
            (0, 1) for low-confidence cell;
            0 for unlabeled pixel, 50-50
            [-1, 0) for background
        '''
        prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        flow = y_hat[:, 1:, :, :]
        weights = y[:, 0, :, :]
        gt_flow = y[:, 1:, :, :]
        
        pos_mask = weights > self.thresh
        if torch.any(weights<0):
            neg_mask = weights < 0
        else: neg_mask = weights == 0
        select_mask = pos_mask or neg_mask

        prob_loss = torch.log(prob) * pos_mask + torch.log(1 - prob) * (neg_mask)
        sum_loss = (self.beta*self.l2(flow, gt_flow) - self.alpha*prob_loss) * select_mask
        if reduction == 'sum':
            sum_loss = torch.sum(sum_loss)
            return sum_loss / torch.sum(select_mask)
        else:
            return sum_loss
    
    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
    #     flow = y_hat[:, 1:, :, :]
    #     return {
    #         'output': prob.detach().cpy().numpy(),
    #         'flow': flow.detach().cpu().numpy()
    #     }