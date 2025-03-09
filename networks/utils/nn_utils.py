import numpy as np

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