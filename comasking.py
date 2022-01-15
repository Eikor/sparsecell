import wandb
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import dataset
from args import describe
import module.comasking as comasking

#### save exp info ###
args = describe('train')

### prepare experiment Material ###
net = comasking.CoMasking(args).cuda()

if args.mode == 'train':
    wandb.init(dir=args.save_dir, config=args)
    train_set = dataset.load_train_dataset(args)
    val_set = dataset.load_val_dataset(args)
    

    ### baseline ###
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    for e in np.arange(args.epochs):
        net.train_epoch(train_loader, e, args)
        stats, masks = net.eval(val_loader, e, args)

### test ###

if args.mode == 'test':
    test_set = dataset.load_test_dataset(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    state_dict = torch.load('result/01_08_02_36_34/epoch_200.pth')
    net.backbone.load_state_dict(state_dict['model_state_dict'])
    stats, masks = net.eval(test_loader, 0, args)
    print(stats)



    

