import argparse
import time

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    # training
    parser.add_argument('--pre_train', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for main task')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--verbose_img', type=str, default=None)
    parser.add_argument('--e1', type=float, default=5)
    parser.add_argument('--e2', type=float, default=5)
    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--start_url', type=str, default=None)
    
    # net
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--coordconv', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    
    # save dir
    parser.add_argument('--save_dir', type=str, default='./result', help='directory to save record file')
    parser.add_argument('--comments', type=str, default='L2')

    # dataset
    parser.add_argument('--annotation_url', type=str, default='./annotations/train')
    parser.add_argument('--train_url', type=str, default='./annotations/train')
    parser.add_argument('--train_url_l', type=str, default='./annotations/train_l')
    parser.add_argument('--train_url_r', type=str, default='./annotations/train_r')
    parser.add_argument('--val_url', type=str, default='./annotations/test')
    parser.add_argument('--test_url', type=str, default='./annotations/test')
    parser.add_argument('--data_url', type=str, default='/home/siat/sdb/datasets/phc_c2c12/090318')
    parser.add_argument('--fulltrain', action='store_true',)
    parser.add_argument('--BEGIN_FRAME', type=int, default=600)
    parser.add_argument('--END_FRAME', type=int, default=700)
    parser.add_argument('--itv', type=int, default=5)
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--rand_flip', action='store_true',)
    parser.add_argument('--rand_crop', action='store_true')
    parser.add_argument('--CROP_SIZE', type=int, default=400)
    parser.add_argument('--resize_ratio', type=float, default=0.5)
    parser.add_argument('--CROP_EDGE', type=int, default=20)
    
    ### test args
    parser.add_argument('--pred_th', type=float, default=0.1)
    parser.add_argument('--patience', type=float, default=5)
    
    parser.add_argument('--ema_alpha', type=float, default=0.8)
    parser.add_argument('--pu_theta', type=float, default=0.3)
    parser.add_argument('--br_alpha', type=float, default=0.1)
    parser.add_argument('--br_gamma', type=float, default=2)
    parser.add_argument('--br_thresh', type=float, default=0.6)
    parser.add_argument('--ece_beta', type=float, default=0.1)
    parser.add_argument('--ece_gamma', type=float, default=2)
    parser.add_argument('--ece_thresh', type=float, default=0.5)
    parser.add_argument('--pose_theta', type=float, default=5)
    parser.add_argument('--comining_thresh', type=float, default=0.6)

    return parser.parse_args()

def get_args():
    args = parse()

    args.comments = args.mode + args.train_url.split('/')[-1]
    args.comments = args.comments + '_' + time.strftime('%m-%d_%H:%M:%S',time.localtime(int(round(time.time()))))
    return args

def describe():
    return get_args()