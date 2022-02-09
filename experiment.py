import wandb
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import dataset
from args import describe
import module
import module.softpose as softpose

#### save exp info ###
args = describe('train')

### prepare experiment Material ###
if args.data_mode =='softpose':
    net = softpose.SoftPose(args).cuda()
else:
    net = module.NN(args).cuda()

# def fft(label_url):
# label_url = ''
# import os
# import matplotlib.pyplot as plt
# fft_flows = []
# fft_sum = []
# fft_highpass = []
# H = np.zeros_like(fft_flows[0])
# ij = np.stack(np.meshgrid(range(704), range(520)))
# ctr = np.array([260, 352])
# dist = np.linalg.norm(ij - ctr.reshape((2, 1, 1)), axis=0)
# R = 20
# H = dist > R
# for label in range(len(os.listdir(label_url))):
#     flow = np.load(os.path.join(label_url, f'{label}.npy'))
#     if len(flow.shape) > 3:
#         flow = flow[0]
#     fft_flow = np.fft.fftshift(np.fft.fft2(flow[1]))
#     fft_flow = np.log(np.abs(fft_flow)**2)
#     fft_sum.append(np.sum(fft_flow))
#     fft_highpass.append(np.sum(fft_flow*H))
#     fft_flows.append(fft_flow)
#     # plt.imshow(fft_flow)
#     # plt.savefig(f'fft/{label}.png')
# fft_highpass = np.array([np.sum(fft_flow*H)/np.sum(H) for fft_flow in fft_flows])
# plt.plot(fft_highpass[worst_case][10:])




    

if args.mode == 'train':
    wandb.init(dir=args.save_dir, config=args)
    train_set = dataset.load_train_dataset(args)
    val_set = dataset.load_val_dataset(args)
    #debug
    # net.backbone.load_state_dict(torch.load('result/01_18_19_50_58/epoch_10.pth')['model_state_dict'])
    ### baseline ###
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, num_workers=4, batch_size=args.batch_size)
    for e in np.arange(args.epochs):
        net.train_epoch(train_loader, e, args)
        stats, masks = net.eval(val_loader, e, args)
    
    # test_set = dataset.load_test_dataset(args)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size)
    # stats, masks = net.eval(test_loader, 0, args)
    # print(np.mean(stats, axis=0))
    # np.savetxt(args.save_dir+'/performance.txt', stats)
### test ###

if args.mode == 'test':
    test_set = dataset.load_test_dataset(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    state_dict = torch.load(args.nn_path)
    net.backbone.load_state_dict(state_dict['model_state_dict'])
    stats, masks = net.eval(test_loader, 0, args)
    print(np.mean(stats, axis=0))
    np.savetxt(args.save_dir+'/performance.txt', stats)
    worst_case = np.argsort(stats[:, -1])
