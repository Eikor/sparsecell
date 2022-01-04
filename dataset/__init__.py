import dataset.detect as detect
import dataset.segment as segment
from torch.utils.data import DataLoader, random_split

data_mode = {
    'point': detect.point,
    'pose': segment.Pose 
}


def load_dataset(args):
    print(f'load {args.dataset}')
    print('-----train-----', end="")
    train_set = data_mode[args.data_mode](args.train_image_url, args.train_anno_url, args)
    print('-----val-----', end='')
    val_set = data_mode[args.data_mode](args.val_image_url, args.val_anno_url, args, Aug=False)
    print('successful Load dataset.')
    
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size)
    return train_set, val_set