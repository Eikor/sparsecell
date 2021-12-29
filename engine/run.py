import torch
import wandb

def train_epoch(net, criterion, optimizer, dataset, epoch, args):
    net.train()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    avg_loss = 0
    # with torch.autograd.detect_anomaly():
    for batch in dataset:
        imgs = batch['image'].to(device=torch.device('cuda'))
        gt = batch['label'].to(device=torch.device('cuda'))

        pred = net(imgs)
        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        avg_loss += loss.item() / len(dataset)
        optimizer.step()

    wandb.log({'train loss': avg_loss.item(), "epoch":epoch})

@torch.no_grad()
def eval_epoch(net, criterion, dataset, epoch, args):
    net.eval()
    avg_loss = 0
    for batch in dataset:
        imgs = batch['image'].to(device=torch.device('cuda'))
        gt = batch['label'].to(device=torch.device('cuda'))

        pred = net(imgs)
        loss = criterion(pred, gt)
        avg_loss += loss.item() / len(dataset)


    wandb.log({'val loss': avg_loss.item(), "epoch":epoch})