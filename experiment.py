import wandb
import torch
import numpy as np
import dataset
from args import describe
import module
from engine import run

#### save exp info ###
args = describe()

### prepare experiment Material ###
args.train_image_url = 'dataset/data/livecell/val/val_images.npy'
args.train_anno_url = 'dataset/data/livecell/val/val_annotation.json'
args.data_mode = 'point'
train_set, val_set = dataset.load_dataset(args)
net = module.NN(args)

### baseline ###

### INCV ###
epochs = np.logspace(args.e1, args.e2, num=args.iter, base=2).astype(int).repeat(2, axis=0)
for epoch in epochs:
    for e in epoch:
        net.train(train_set, e, args)
        net.eval(val_set, e, args)


### test ###