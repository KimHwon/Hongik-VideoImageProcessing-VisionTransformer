import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from task import train, inference, validate
from task import load_checkpoint


def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
