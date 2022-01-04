import argparse
import time
import os

def parse():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--epochs', type=int, default=200, help='epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for main task')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=10)
    
    # net
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--pose_theta', type=float, default=5)

    # save dir
    parser.add_argument('--save_dir', type=str, default='./result', help='directory to save record file')

    # dataset
    parser.add_argument('--dataset', type=str, default='livecell')
    parser.add_argument('--data_mode', type=str, default='pose')
    parser.add_argument('--anno_rate', type=float, default=0.1)
    parser.add_argument('--crop_size', type=int, default=400)

    
    ### test args
    parser.add_argument('--pred_th', type=float, default=0.1)
    parser.add_argument('--patience', type=float, default=5)
    

    return parser.parse_args()

def get_args():
    num_channels = {
        'livecell': 1,
        'tissuenet': 2,
    }
    segmentation = ['pose']
    detection = ['point']
    args = parse()
    args.train_image_url = f'./dataset/data/{args.dataset}/train/train_images.npy'
    args.train_anno_url = f'./dataset/data/{args.dataset}/train/train_annotation_{args.anno_rate}.npz'
    args.test_image_url = f'./dataset/data/{args.dataset}/train/train_images.npy'
    args.test_anno_url = f'./dataset/data/{args.dataset}/train/train_annotation_{args.anno_rate}.npz'
    args.val_image_url = f'./dataset/data/{args.dataset}/val/val_images.npy'
    args.val_anno_url = f'./dataset/data/{args.dataset}/val/val_annotation.npz'

    args.num_channels = num_channels[args.dataset]
    
    if args.data_mode in segmentation:
        print(f'using {args.data_mode} to segment {args.dataset}')
        if args.data_mode == 'pose':
            args.num_classes = 3
    elif args.data_mode in detection:
        print(f'using {args.data_mode} to detect {args.dataset}')
    else:
        pass

    args.save_dir = 'result/' + time.strftime('%m_%d_%H_%M_%S',time.localtime(int(round(time.time()))))
    try:
        os.makedirs(args.save_dir)
        print(f'save experiment result in {args.save_dir}')
    except:
        pass
    return args

def describe():
    return get_args()