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
        self.theta = args.pose_theta
        self.thresh = 0.5
        self.l2 = nn.MSELoss(reduce=None)
   
    def forward(self, y_hat, y):
        weights = y[:, 0, :, :]
        mask = weights > self.thresh
        prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        flow = y_hat[:, 1:, :, :]
        gt_flow = y[:, 1:, :, :]
        prob_loss = torch.log(prob) * mask + torch.log(1 - prob) * (~mask)
        return torch.mean(weights * self.theta*self.l2(flow, gt_flow) - prob_loss)
    
    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
    #     flow = y_hat[:, 1:, :, :]
    #     return {
    #         'output': prob.detach().cpy().numpy(),
    #         'flow': flow.detach().cpu().numpy()
    #     }