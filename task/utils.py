import logging
import time
import pathlib

def get_logger(name=None, path=None):
    path = path if path else f'logs/{time.ctime()}.log'
    path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    
    stream = logging.StreamHandler()
    stream.setFormatter(
        logging.Formatter("%(message)s")
    )
    logger.addHandler(stream)

    file = logging.FileHandler(filename=path)
    file.setFormatter(
        logging.Formatter("%(asctime)s | [%(levelname)s] | {%(filename)s:%(lineno)d} | %(message)s")
    )
    logger.addHandler(file)

    return logger


def load_checkpoint(args):
    pass
    