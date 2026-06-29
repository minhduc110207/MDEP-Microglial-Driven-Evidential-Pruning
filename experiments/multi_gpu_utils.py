import torch
import torch.nn as nn

class TransparentDataParallel(nn.DataParallel):
    """
    A wrapper around nn.DataParallel that forwards attribute access to the underlying module.
    This allows drop-in multi-GPU support without breaking code that accesses model.fc, model.backbone, etc.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
            
    def __setattr__(self, name, value):
        if name in ["module", "device_ids", "output_device", "dim", "_is_replica"]:
            super().__setattr__(name, value)
        elif hasattr(self, "module") and hasattr(self.module, name):
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)
