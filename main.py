import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import PIL

from model import VisionTransformer as ViT
from model import select_configs
from task import train, inference, validate
from task import get_logger


_logger = get_logger('ViT')

try:
    from apex.parallel import DistributedDataParallel
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    _logger.exception("Please install APEX from https://www.github.com/nvidia/apex to run this example.")
    exit(1)

def main(args):
    # Print configurations.
    for k, v in vars(args).items():
        _logger.info(f" {k:<25} : {v}")


    cudnn.benchmark = True
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    

    config = select_configs(args.arch)
    for key, value in config.items():
        if value:
            config[key] = value
    model = ViT(**config)
    _logger.info(f"Using model '{args.arch}' (pretrained={args.pretrained})")


    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    model = model.cuda().to(memory_format=memory_format)

    if args.sync_bn:
        import apex
        _logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)


    # Initalize AMP.
    if args.task == 'inference':
        model = amp.initialize(
            model,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale
        )
    else:   # train, validate
        args.lr = args.lr * float(args.batch_size * args.world_size) / 256.0
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale
        )
    
    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    

    criterion = nn.CrossEntropyLoss().cuda()


    # Resume from checkpoint.
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            _logger.info(f"> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.arch = checkpoint['arch']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            _logger.info(f"=> Loaded checkpoint '{args.arch}' (epoch {args.start_epoch})")
        else:
            _logger.info(f"=> No checkpoint found at '{args.resume}'")


    # Load datasets.
    train_loader, train_sampler = load_train_dataset('train', args)
    validate_loader, validate_sampler = load_validate_dataset('val', args)


    if args.task == 'train':
        train(train_sampler, train_loader, validate_loader, model, criterion, optimizer, amp, args)
    elif args.task == 'validate':
        loss, prec1, prec5 = validate(validate_loader, model, criterion)
        _logger.info(f"-==| Loss: {loss} | Prec@1: {prec1} | Prec@5: {prec5} |==-")
    elif args.task == 'inference':
        pass
    else:
        _logger.exception(f"Unknown task '{args.task}'")
        exit(1)


def load_train_dataset(train_subdir, args):
    train_dataset = datasets.ImageFolder(
        os.path.join(args.dataset, train_subdir),
        transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    return train_loader, train_sampler

def load_validate_dataset(validate_subdir, args):
    validate_dataset = datasets.ImageFolder(
        os.path.join(args.dataset, validate_subdir),
        transforms.Compose([
            transforms.Resize(args.image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    validate_sampler = None
    if args.distributed:
        validate_sampler = DistributedSampler(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=validate_sampler
    )
    
    return validate_loader, validate_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Vision Transformer')
    parser.add_argument('task', choices=['train', 'validate', 'inference'],
                        help='task that model to do')

    parser.add_argument('dataset',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', default='B_16',
                        help='model architecture (default: B_16)')

    parser.add_argument('-e', '--epochs', default=10000, type=int,
                        help='number of total epochs to run (default: 10000)')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        help='mini-batch size per process (default: 1024)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')

    # Model hyper-parameters.
    parser.add_argument('--image-size', default=224, type=int,
                        help='image size (default: 224)')
    parser.add_argument('--patch-size', type=int,
                        help='patch size')
    parser.add_argument('--num-heads', type=int,
                        help='number of heads')
    parser.add_argument('--num-layers', type=int,
                        help='number of layers')
    parser.add_argument('--embed-size', type=int,
                        help='Embed size')
    parser.add_argument('--mlp-size', type=int,
                        help='MLP size')
    parser.add_argument('--dropout-rate', type=float,
                        help='dropout rate')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--debug', action='store_true',
                        help='debug memory usages.')

    # Arguments for APEX.
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    main(args)
