import logging
from datetime import datetime
import pathlib


__TIMESTAMP__ = None
def get_timestamp():
    global __TIMESTAMP__
    if __TIMESTAMP__ is None:
        __TIMESTAMP__ = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    return __TIMESTAMP__

def get_logger(name=None, path=None):
    path = path if path else f'logs/{get_timestamp()}.log'
    path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.Logger(name)
    logger.propagate = False
    
    stream = logging.StreamHandler()
    stream.setFormatter(
        logging.Formatter("%(message)s")
    )
    stream.setLevel(logging.INFO)
    logger.addHandler(stream)

    file = logging.FileHandler(filename=path)
    file.setFormatter(
        logging.Formatter("%(asctime)s | [%(levelname)s] | {%(filename)s:%(lineno)d} | %(message)s")
    )
    file.setLevel(logging.DEBUG)
    logger.addHandler(file)

    return logger


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.last = ''
        self.sum = 0.0
        self.cnt = 0
    
    def update(self, value, step=1):
        self.last = value
        self.sum += value
        self.cnt += step

    def get_average(self):
        return self.sum / self.cnt

    def __str__(self):
        return f"{self.name} {self.last:.6f} ({self.get_average():.3f})"

class ProgressMeter(object):
    def __init__(self, name, count, meters):
        self.name = name
        self.step = 0
        self.count = count
        self.len = len(str(count))
        self.meters = meters
    
    def update(self, step=1):
        self.step += step
        if self.step > self.count:
            self.step = self.count
    
    def __str__(self):
        entries = '\t'.join([str(meter) for meter in self.meters])
        progress = str(self.step).rjust(self.len, ' ')
        return f"{self.name} [{progress}/{self.count}] {entries}"
