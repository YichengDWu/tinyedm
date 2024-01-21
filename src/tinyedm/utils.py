import inspect
import torch


def deinstantiate(obj):
    class_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    init_signature = inspect.signature(obj.__init__)
    init_params = init_signature.parameters

    params = {}
    for name, param in init_params.items():
        if name == "self":
            continue

        if hasattr(obj, name):
            attribute = getattr(obj, name)

            if isinstance(attribute, torch.nn.Module):
                params[name] = deinstantiate(attribute)
            elif isinstance(attribute, tuple):
                params[name] = list(attribute)
            else:
                params[name] = attribute
        elif param.default is not inspect.Parameter.empty:
            params[name] = param.default

    return {"_target_": class_name, **params}


def swap_tensors(tensor1, tensor2):
    tmp = torch.empty_like(tensor1)
    tmp.copy_(tensor1)
    tensor1.copy_(tensor2)
    tensor2.copy_(tmp)
