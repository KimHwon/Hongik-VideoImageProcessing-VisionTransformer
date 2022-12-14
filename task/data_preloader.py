import torch

from .utils import get_logger


_logger = get_logger('ViT')

class DataPreloader():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        except Exception as ex:
            _logger.error(f"Error while prefetch dataset : {str(ex)}")
            self.next_input = None
            self.next_target = None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        target = self.next_target
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        self.preload()
        
        return input, target
    
    def __iter__(self):
        return self
    
    def __next__(self):
        input, target = self.next()
        if input is None or target is None:
            raise StopIteration
        else:
            return input, target