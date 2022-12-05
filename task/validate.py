import torch
import torch.distributed as dist

from apex.fp16_utils import to_python_float

from .data_preloader import DataPreloader
from .utils import get_logger
from .utils import AverageMeter, ProgressMeter


_logger = get_logger('ViT')

def validate(loader, model, criterion, args):
    loss_metric = AverageMeter('Loss')
    prec1_metric = AverageMeter('Prec@1')
    prec5_metric = AverageMeter('Prec@5')
    progress = ProgressMeter(f"Validate : ", len(loader), [loss_metric, prec1_metric, prec5_metric])

    model.eval()
    loader = DataPreloader(loader)

    for idx, sample in enumerate(loader):
        input, target = sample

        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, top_k=(1, 5))
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)
        else:
            reduced_loss = loss.data
        

        loss_metric.update(to_python_float(reduced_loss), input.size(0))
        prec1_metric.update(to_python_float(prec1), input.size(0))
        prec5_metric.update(to_python_float(prec5), input.size(0))
        progress.update(1)

        if args.local_rank == 0 and idx % 10 == 0:
            _logger.info(str(progress))

    return (
        loss_metric.get_average(),
        prec1_metric.get_average(),
        prec5_metric.get_average()
    )


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt
