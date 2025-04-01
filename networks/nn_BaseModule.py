import torch
import torch.nn as nn
import numpy as np

class BaseModule(nn.Module):
    def __init__(self, **args):
        super(BaseModule, self).__init__(**args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def shape_image(self, image):
        return torch.from_numpy(np.pad(np.log(image+1),(0,0))).float().unsqueeze(1).to(self.device)

    def shape_data(self, batch, target_index=1):
        input_tensor = self.shape_image(np.array([b[0] for b in batch]))
        target_tensor = self.shape_image(np.array([b[target_index] for b in batch]))
        return input_tensor, target_tensor