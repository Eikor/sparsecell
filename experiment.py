import wandb
import torch
import numpy as np
import dataset
from args import describe
import module

#### save exp info ###
args = describe()

### prepare experiment Material ###
train_set, val_set = dataset.load_dataset(args)
net, criterion = module.load_module(args)

### baseline ###

### INCV ###
epochs = np.logspace(args.e1, args.e2, num=args.iter, base=2).astype(int).repeat(2, axis=0)
### test ###