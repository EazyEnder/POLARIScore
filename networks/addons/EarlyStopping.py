from typing import Dict, Union

class EarlyStopping:
    def __init__(self, patience:int=5, delta:float=0., warmup:int=100):
        self.patience:int = patience
        """Time frame needed to stop the training if the model isn't better."""
        self.delta:float = delta
        self.min_loss:float = None
        self.early_stop:bool = False
        """If true, the training will stop"""
        self.counter:int = 0
        """Times (x epochs) when the model is consecutively worse than the best one."""
        self.warmup:int = warmup
        """Early stop can be applied only after epochs >= warmup"""

    def __call__(self, val_loss, epoch:Union[int,None]=None):

        if epoch is not None and epoch < self.warmup:
            return
        
        def _set():
            self.min_loss = val_loss

        if self.min_loss is None:
            _set()
        elif val_loss > self.min_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            _set()
            self.counter = 0