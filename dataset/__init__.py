import dataset.detect as detect
import dataset.segment as segment
from torch.utils.data import DataLoader, random_split




def load_dataset(args):
    if args.data_mode == 'point':
        train_set = detect.point(args.train_image_url, args.train_anno_url, args)
        val_set = detect.point(args.val_image_url, args.val_anno_url, args)
    elif args.data_mode == 'pose':
        train_set = segment.pose(args.train_image_url, args.train_anno_url, args)
        val_set = segment.pose(args.val_image_url, args.val_anno_url, args)
    
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    return train_loader, val_loader