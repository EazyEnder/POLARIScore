import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import get_type_hints


#TODO

def is_a_module(module):
    cls = module if str(module.__class__) == "<class 'type'>" else module.__class__
    if issubclass(cls, nn.Module):
        return True
    return False

def _extract_layer_params(layer):
    params = {}
    try:
        for k, v in layer.__dict__.items():
            if not k.startswith('_') and isinstance(v, (int, float, tuple)):
                params[k] = v
    except:
        pass
    return params  

def serialize_module(module_class, _name=None, *init_args):
    init_sig = inspect.signature(module_class.__init__)
    type_hints = get_type_hints(module_class.__init__)

    args_info = []
    for name, param in init_sig.parameters.items():
        if name == "self":
            continue
        arg_type = type_hints.get(name, "Any")
        arg_type_str = arg_type.__name__ if hasattr(arg_type, "__name__") else str(arg_type)
        default_val = param.default if param.default is not inspect.Parameter.empty else None

        args_info.append({
            "name": name,
            "type": arg_type_str,
            "default": default_val
        })

    output = {
        "class": module_class.__name__ if _name is None else _name,
        "args": args_info,
        "layers": [],
    }

    if not(is_a_module(module_class)):
        return {"name": name,
        "type": module_class.__name__,
        "params": _extract_layer_params(module_class)}
        
    model = module_class(*init_args)      

    for name, module in model.named_modules():
        if name == "":
            continue
        
        if isinstance(module, nn.Sequential):
            for idx, submodule in enumerate(module):
                sub_layer_info = serialize_module(submodule.__class__, *[])
                sub_layer_info["name"] = f"{name}.{idx}"
                output["layers"].append(sub_layer_info)
        if is_a_module(module):
            layer_info = serialize_module(module, _name=name)
            output["layers"].append(layer_info)
        else:
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "params": _extract_layer_params(module)
            }
            output["layers"].append(layer_info)

    return output

if __name__ == "__main__":
    import os
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    from networks.nn_UNet import ConvBlock
    info = serialize_module(ConvBlock)
    print(json.dumps(info, indent=2))