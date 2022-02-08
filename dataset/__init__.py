import dataset.detect as detect
import dataset.segment as segment
from torch.utils.data import DataLoader, random_split

data_mode = {
    'point': detect.point,
    'pose': segment.Pose,
    'softpose': segment.SoftPose 
}

def load_train_dataset(args):
    print(f'load {args.dataset}')
    print('-----train-----')
    train_set = data_mode[args.data_mode](args.train_image_url, args.train_anno_url, args.train_label_url, args)
    print('-------------')
    return train_set

def load_val_dataset(args):
    print(f'load {args.dataset}')
    print('-----val-----')
    val_set = data_mode[args.data_mode](args.val_image_url, args.val_anno_url, args.val_label_url, args, Aug=False)
    print('-------------')
    return val_set

def load_test_dataset(args):
    print(f'load {args.dataset}')
    print('-----test-----')
    test_set = data_mode[args.data_mode](args.test_image_url, args.test_anno_url, args.test_label_url, args, Aug=False)
    print('-------------')
    return test_set