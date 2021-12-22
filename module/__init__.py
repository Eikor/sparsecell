import loss_fn
import unet

def load_module(args):
    net = unet.UNet(args.num_channels, args.num_classes)
    criterion = loss_fn.PoseLoss(args)
    return net, criterion