import numpy as np

from torch.nn import init
def xavier_init(model):
    for layer in model.modules():
        if hasattr(layer, 'initialize') and callable(layer.initialize):
            continue
        if hasattr(layer, 'weight') and layer.weight is not None:
            if layer.weight.ndim >= 2:
                init.xavier_uniform_(layer.weight)
            else:
                init.ones_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)

def compute_accuracy(label, pred,sigma=.1,log10=True):
    if log10:
        pred = np.log(pred)/np.log(10)
        label = np.log(label)/np.log(10)
    corrects = (np.abs(pred-label) <= sigma)
    acc = corrects.sum() / (corrects.shape[0]*corrects.shape[1])
    return acc

def compute_batch_accuracy(batch,sigma=.1,log10=True):
    acc = []
    for label, pred in batch:
        acc.append(compute_accuracy(label, pred, sigma, log10))
    return (np.mean(acc),np.std(acc))