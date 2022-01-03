import dataset.detect as detect
import dataset.segment as segment
from torch.utils.data import DataLoader, random_split




def load_dataset(args):
    print(f'load {args.dataset}')
    print('     ', end=None)
    if args.data_mode == 'point':
        print('train---     ', end=None)
        train_set = detect.point(args.train_image_url, args.train_anno_url, args)
        print('val---     ', end=None)
        val_set = detect.point(args.val_image_url, args.val_anno_url, args)
    elif args.data_mode == 'pose':
        print('train---     ', end=None)
        train_set = segment.Pose(args.train_image_url, args.train_anno_url, args)
        print('val---     ', end=None)
        val_set = segment.Pose(args.val_image_url, args.val_anno_url, args)
    print('Done.')
    
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size)
    return train_set, val_set