import torch

from apex.fp16_utils import to_python_float

from .validate import validate
from .validate import accuracy, reduce_tensor
from .data_preloader import DataPreloader
from .utils import get_timestamp, get_logger
from .utils import AverageMeter, ProgressMeter


_logger = get_logger('ViT')

def train(train_sampler, train_loader, validate_loader, model, criterion, optimizer, amp, args):
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, amp, args)

        loss, prec1, prec5 = validate(validate_loader, model, criterion, args)
        _logger.info(f"-==| Loss: {loss} | Prec@1: {prec1} | Prec@5: {prec5} |==-")

        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            
            torch.save({
                'epoch': epoch+1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, f"checkpoint_{get_timestamp()}.pth.tar")
            if is_best:
                import shutil
                shutil.copy(f"checkpoint_{get_timestamp()}.pth.tar", f"checkpoint_{get_timestamp()}_Acc.{best_prec1:3}.pth.tar")

def train_one_epoch(loader, model, criterion, optimizer, epoch, amp, args):
    loss_metric = AverageMeter('Loss')
    prec1_metric = AverageMeter('Prec@1')
    prec5_metric = AverageMeter('Prec@5')
    progress = ProgressMeter(f"Train : [{epoch}]", len(loader), [loss_metric, prec1_metric, prec5_metric])

    model.train()
    loader = DataPreloader(loader)

    for idx, sample in enumerate(loader):
        input, target = sample

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

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

        torch.cuda.synchronize()
        if args.local_rank == 0 and idx % 10 == 0:
            _logger.info(str(progress))
