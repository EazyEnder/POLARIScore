from astropy import units as u
import os
import numpy as np
from config import *
import inspect
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from kan import KAN

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from training_batch import open_batch
batch = open_batch("batch_bb133abb-ef3f-4f17-a504-2eae88d6544b")
training_inputs = np.array([b[0].flatten() for b in batch])
training_outputs = np.array([b[1].flatten() for b in batch])

shape = training_inputs.shape[1]
print(f"Training Inputs Shape: {training_inputs.shape}")
print(f"Training Outputs Shape: {training_outputs.shape}")

training_inputs = torch.from_numpy(training_inputs)
training_outputs = torch.from_numpy(training_outputs)
from kan.utils import create_dataset_from_data
dataset = create_dataset_from_data(training_inputs,training_outputs, device=device)

"""
model = KAN(width=[shape,20,shape], grid=5, k=3, seed=1, device=device)
results = model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
"""

model = KAN.loadckpt('./model/' + '0.1')
model.refine(10)
results = model.fit(dataset, opt="LBFGS", steps=50)
train_losses = results['train_loss']
test_losses = results['test_loss']
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()


"""
from training_batch import open_batch
batch = open_batch("batch_4a20fa23-4833-434d-b06a-72589b5201d6")
test_inputs = np.array([b[0].flatten() for b in batch])
test_outputs = np.array([b[1].flatten() for b in batch])
predictions = model(torch.from_numpy(test_inputs[0]).unsqueeze(0))
print(predictions)
"""

plt.show()