#Exponential Moving Average 
import copy
import torch
import torch.nn as nn
from POLARIScore.config import LOGGER

class ExponentialMovingAverage:
    """
    Maintains an exponential moving average (EMA) of model parameters.
    """

    def __init__(self, momentum: float = 0.999):
        """
        Args:
            momentum (float): The EMA momentum (typical values: 0.99-0.9999).
        """
        self.momentum = momentum
        self.ema_parameters = {}

    def register_model(self, model: nn.Module):
        """Initialize EMA parameters from the given model."""
        LOGGER.log("EMA enabled on model")
        for param_name, param_tensor in model.named_parameters():
            if param_tensor.requires_grad:
                self.ema_parameters[param_name] = param_tensor.detach().clone().to(param_tensor.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the EMA parameters using the current model parameters.
        EMA: shadow = momentum * shadow + (1 - momentum) * current
        """
        for param_name, param_tensor in model.named_parameters():
            if param_tensor.requires_grad:
                if param_name not in self.ema_parameters:
                    raise KeyError(f"Parameter {param_name} was not registered in EMA.")
                shadow_param = self.ema_parameters[param_name]
                shadow_param.mul_(self.momentum).add_(param_tensor.detach(), alpha=1 - self.momentum)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """Replace model parameters with their EMA counterparts (in-place)."""
        for param_name, param_tensor in model.named_parameters():
            if param_tensor.requires_grad and param_name in self.ema_parameters:
                param_tensor.copy_(self.ema_parameters[param_name])

    def copy_ema_model(self, model: nn.Module) -> nn.Module:
        """
        Create and return a deep-copied version of the model with EMA weights applied.
        """
        ema_model_copy = copy.deepcopy(model)
        self.apply_to(ema_model_copy)
        return ema_model_copy

    def state_dict(self) -> dict:
        """Return a state dict containing the EMA shadow parameters."""
        return {name: tensor.clone() for name, tensor in self.ema_parameters.items()}

    def load_state_dict(self, state_dict: dict):
        """Load EMA shadow parameters from a state dict."""
        self.ema_parameters = {name: tensor.clone() for name, tensor in state_dict.items()}