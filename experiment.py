import wandb
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import dataset
from args import describe
import module

#### save exp info ###
args = describe()

### prepare experiment Material ###
wandb.init(dir=args.save_dir, config=args)
train_set, val_set = dataset.load_dataset(args)

### baseline ###
net = module.NN(args).cuda()
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)

for e in np.arange(args.epochs):
    net.train(train_loader, e, args)
    stats, masks = net.eval(val_loader, e, args)


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
