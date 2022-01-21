import argparse
import time
import os

def parse():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--epochs', type=int, default=70, help='epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for main task')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--save_interval', type=int, default=10)
    
    # net
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--pose_loss', type=str, default='l1')
    parser.add_argument('--pose_alpha', type=float, default=1)
    parser.add_argument('--pose_beta', type=float, default=5)
    parser.add_argument('--consist', type=float, default=0.1)
    parser.add_argument('--maskunlabel', action='store_true')
    parser.add_argument('--nonlinear_flow', action='store_true')
    parser.add_argument('--flow_c', type=float, default=1)
    parser.add_argument('--neg_ratio', type=float, default=0.2)
    parser.add_argument('--pos_ratio', type=float, default=0.1)



    # save dir
    parser.add_argument('--save_dir', type=str, default='./result', help='directory to save record file')

    # dataset
    parser.add_argument('--dataset', type=str, default='livecell')
    parser.add_argument('--data_mode', type=str, default='pose')
    parser.add_argument('--anno_rate', type=float, default=0.1)
    parser.add_argument('--crop_size', type=int, default=320)

    
    ### test args
    parser.add_argument('--test_url', type=str, default=None)

    return parser.parse_args()

def get_args(mode):
    num_channels = {
        'livecell': 1,
        'tissuenet': 2,
    }
    segmentation = ['pose']
    detection = ['point']
    args = parse()
    args.mode = mode

    args.train_image_url = f'./dataset/data/{args.dataset}/train/train_images.npy'
    args.train_anno_url = f'./dataset/data/{args.dataset}/train/train_annotation_{args.anno_rate}.npz'
    args.train_label_url = f'./dataset/data/{args.dataset}/train/{args.data_mode}_label{args.anno_rate}'
    args.test_image_url = f'./dataset/data/{args.dataset}/test/test_images.npy'
    args.test_anno_url = f'./dataset/data/{args.dataset}/test/test_annotation.npz'
    args.test_label_url = f'./dataset/data/{args.dataset}/test/{args.data_mode}_label'
    args.val_image_url = f'./dataset/data/{args.dataset}/val/val_images.npy'
    args.val_anno_url = f'./dataset/data/{args.dataset}/val/val_annotation.npz'
    args.val_label_url = f'./dataset/data/{args.dataset}/val/{args.data_mode}_label'
    
    args.num_channels = num_channels[args.dataset]
    
    if args.data_mode in segmentation:
        print(f'using {args.data_mode} to segment {args.dataset}')
        if args.data_mode == 'pose':
            args.num_classes = 3
    elif args.data_mode in detection:
        print(f'using {args.data_mode} to detect {args.dataset}')
    else:
        pass

    if args.mode == 'test':
        assert args.test_url is not None
        args.save_dir = args.test_url
        args.nn_path = os.path.join(args.save_dir, 'epoch_70.pth')
    else:
        args.save_dir = 'result/' + time.strftime('%m_%d_%H_%M_%S',time.localtime(int(round(time.time()))))
    try:
        os.makedirs(args.save_dir)
        print(f'save experiment result in {args.save_dir}')
    except:
        print(f'save experiment result in {args.save_dir}')
    return args

def describe(mode):
    args = get_args(mode)
    # args.mode = mode
    return args