import wandb
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import dataset
from args import describe
import module

#### save exp info ###
args = describe('train')

### prepare experiment Material ###
net = module.NN(args).cuda()


if args.mode == 'train':
    wandb.init(dir=args.save_dir, config=args)
    train_set = dataset.load_train_dataset(args)
    val_set = dataset.load_val_dataset(args)
    

    ### baseline ###
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    for e in np.arange(args.epochs):
        net.train(train_loader, e, args)
        stats, masks = net.eval(val_loader, e, args)

### test ###

if args.mode == 'test':
    test_set = dataset.load_test_dataset(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    state_dict = torch.load('result/01_08_02_36_34/epoch_200.pth')
    net.backbone.load_state_dict(state_dict['model_state_dict'])
    stats, masks = net.eval(test_loader, 0, args)
    print(stats)

### INCV ###
# random split dataset into 2 folds
# train_set1, train_set2 = random_split(train_set, 0.5)
# train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True)
# train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True)
# # build two networks
# net1 = module.NN(args)
# net2 = module.NN(args)


# epochs = np.logspace(args.e1, args.e2, num=args.iter, base=2).astype(int).repeat(2, axis=0)
# for epoch in epochs:
#     for e in epoch:
#         net1.train(train_loader1, e, args)
#         net.eval(val_set, e, args)


### test ###
