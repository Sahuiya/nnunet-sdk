import torch
from typing import Any, Union


def load_nnunet_checkpoint(
    filename: str,
    map_location: Union[torch.device, str, None] = None,
) -> Any:
    """
    Load a nnU-Net training checkpoint (.pth).

    PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects nnU-Net
    checkpoints that store numpy scalars and other metadata. nnU-Net checkpoints
    are trusted local training artifacts, so weights_only=False is used.
    """
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    try:
        return torch.load(filename, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(filename, **kwargs)


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
