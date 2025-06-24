"""
Utilities for working with PyTorch .pt files
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def save(obj: Any, path: Union[str, Path], **kwargs) -> None:
    """Save a PyTorch object to a .pt file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path, **kwargs)


def load(path: Union[str, Path], map_location: Optional[str] = None, **kwargs) -> Any:
    """Load a PyTorch object from a .pt file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return torch.load(path, map_location=map_location, **kwargs)


def save_model(model: torch.nn.Module, path: Union[str, Path], save_full_model: bool = False) -> None:
    """Save a PyTorch model (state dict by default, full model if specified)"""
    if save_full_model:
        save(model, path)
    else:
        save(model.state_dict(), path)


def load_model(
    model_class: torch.nn.Module, path: Union[str, Path], map_location: Optional[str] = None, strict: bool = True
) -> torch.nn.Module:
    """Load model weights into a model instance"""
    state_dict = load(path, map_location=map_location)
    model_class.load_state_dict(state_dict, strict=strict)
    return model_class


def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a .pt file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Load without mapping to get basic info
    obj = load(path, map_location="cpu")

    info = {"file_size_mb": path.stat().st_size / (1024 * 1024), "type": type(obj).__name__}

    # Add tensor-specific info
    if isinstance(obj, torch.Tensor):
        info.update({"shape": obj.shape, "dtype": obj.dtype, "device": obj.device, "requires_grad": obj.requires_grad})

    # Add state dict info
    elif isinstance(obj, dict):
        info.update(
            {
                "num_parameters": len(obj),
                "parameter_names": list(obj.keys())[:10],  # First 10 keys
            }
        )

    return info


def verify_model_compatibility(model: torch.nn.Module, state_dict_path: Union[str, Path]) -> bool:
    """Check if a state dict is compatible with a model"""
    try:
        state_dict = load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception:
        return False
