import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from task import train, inference, validate

try:
    from apex.parallel import DistributedDataParallel
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install APEX from https://www.github.com/nvidia/apex to run this example.")



def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Vision Transformer')
    parser.add_argument('task', choices=['train', 'validate', 'inference']
                        help='task that model to do')

    parser.add_argument('dataset',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', default='B_16'
                        help='model architecture (default: B_16)')

    parser.add_argument('-e', '--epochs', default=1000, type=int,
                        help='number of total epochs to run (default: 1000)')
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
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size (default: 224)')

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    # Arguments for APEX.
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    main(args)
