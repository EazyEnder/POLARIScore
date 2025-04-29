import torch
import torch.nn as nn
import numpy as np
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config import LOGGER

class BaseModule(nn.Module):
    def __init__(self, **args):
        super(BaseModule, self).__init__(**args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def shape_image(self, image):
        if np.min(image) > 0:
            image = np.log(image)
        return torch.from_numpy(image).float().unsqueeze(1).to(self.device)

    def shape_data(self, batch, target_indexes=1, input_indexes=0):
        
        assert (len(batch)) > 0, LOGGER.error("Can't apply the model on the batch, the batch is empty.")

        input_indexes = input_indexes if type(input_indexes) is list else [input_indexes]
        input_tensors = None
        for t in input_indexes:
            if input_tensors is None:
                input_tensors = [self.shape_image(np.array([b[t] for b in batch]))]
            else:
                input_tensors.append(self.shape_image(np.array([b[t] for b in batch])))

        target_indexes = target_indexes if type(target_indexes) is list else [target_indexes]
        target_tensors = None
        for t in target_indexes:
            if target_tensors is None:
                target_tensors = [self.shape_image(np.array([b[t] for b in batch]))]
            else:
                target_tensors.append(self.shape_image(np.array([b[t] for b in batch])))

        return input_tensors, target_tensors
    
    def forward(self, x):
        if type(x) is list:
            if len(x) > 1:
                return torch.cat(x, dim=1)
            else:
                return x[0]
        return x
