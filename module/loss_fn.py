import torch
import torch.nn as nn

class BRLoss(nn.Module):
    def __init__(self, args):
        super(BRLoss, self).__init__()
        self.alpha = args.br_alpha
        self.gamma = args.br_gamma
        self.threshold = args.br_thresh

    def forward(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat).clamp(0.0001, 0.9999)
        calib_mask = (y_hat > self.threshold) * (1-y) # "hard negative in negatives(1-y)"
        pos_loss = torch.pow((1 - y_hat), self.gamma) * torch.log(y_hat) * y
        neg_loss = torch.pow(y_hat, self.gamma) * torch.log(1-y_hat) * (1 - calib_mask) * (1 - y) \
            + torch.pow((1 - y_hat), self.gamma) * torch.log(y_hat) * calib_mask
        return -torch.mean(pos_loss + self.alpha * neg_loss)


class PULoss(nn.Module):
    def __init__(self, args):
        super(PULoss, self).__init__()
        self.theta = args.pu_theta

    def forward(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat).clamp(0.0001, 0.9999)
        pos_loss = -self.theta * torch.log(y_hat) * y
        neg_mask = y<0.01
        neg_loss = torch.clamp(-torch.log(1-y_hat) * neg_mask + self.theta * torch.log(1-y_hat) * y, min=0)
        # print(f'pos:{torch.mean(pos_loss).item()} neg:{torch.mean(neg_loss).item()}')
        return torch.mean(pos_loss + neg_loss)


class clsloss(nn.Module):
    def __init__(self):
        super(clsloss, self).__init__()
    def forward(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat).clamp(min=1e-4, max=(1 - 1e-4))
        # if torch.any(y_hat == 0) or torch.any(y_hat == 1):
        #     print('error')
        pos_loss = torch.log(y_hat) * y
        neg_mask = y < 0.1
        neg_loss = torch.log(1 - y_hat) * neg_mask
        # print(f'{torch.mean(pos_loss)}, {torch.mean(neg_loss)}')
        return -torch.mean(pos_loss + neg_loss)  

class eceloss(nn.Module):
    def __init__(self, args):
        super(eceloss, self).__init__()
        self.gamma = args.ece_gamma
        self.threshold = args.ece_thresh
        self.beta = args.ece_beta

    def forward(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat).clamp(0.0001, 0.9999)
        calib_mask = (y_hat > self.threshold) * (1-y) # "hard negative in negatives(1-y)"
        pos_loss = torch.pow((1 - y_hat), self.gamma) * torch.log(y_hat) * y
        neg_loss = torch.pow(y_hat, self.gamma) * torch.log(1-y_hat) * (1 - calib_mask) * (1 - y) 
        return -torch.mean(pos_loss + neg_loss)
        

class PoseLoss(nn.Module):
    def __init__(self, args):
        super(PoseLoss, self).__init__()
        self.theta = args.pose_theta
        self.l2 = nn.MSELoss()
    def forward(self, y_hat, y):
        prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        flow = y_hat[:, 1:, :, :]
        gt_prob = y[:, 0, :, :]
        gt_flow = y[:, 1:, :, :]
        prob_loss = -torch.mean(torch.log(prob) * gt_prob + torch.log(1 - prob) *(1-gt_prob))
        return self.theta*self.l2(flow, gt_flow) + prob_loss