# refer to: https://github.com/plae-tljg/GPT-SoVITS-Musa/blob/main/code_patches.zip
# Kakaru(https://github.com/KakaruHayate/) 2025/10/11

import logging
import os
import shutil
import subprocess
from typing import Any, Optional, Union

import torch
import torch_musa
from typing_extensions import override

import pytorch_lightning as pl
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import logging
_log = logging.getLogger(__name__)

class MUSAAccelerator(Accelerator):
    """Accelerator for Moore Threads MUSA devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException: If the selected device is not MUSA.
        """
        if device.type != "musa":
            raise MisconfigurationException(f"Device should be MUSA, got {device} instead.")
        if device.index is None:
            device = torch.device(f"musa:0")
        torch.musa.set_device(device)

    #@override
    #def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
    #    if device.type == "musa":
    #        return torch_musa.memory_stats(device)

    @override
    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""
        torch.musa.empty_cache()

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return torch.device("musa") if _musa_available else None

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, torch.device):
            return [devices]
        if devices is None or devices == "auto":
            num_devices = MUSAAccelerator.auto_device_count()
            devices = list(range(num_devices))
        elif isinstance(devices, int):
            devices = list(range(devices))
        elif isinstance(devices, (list, tuple)):
            pass
        else:
            raise ValueError(f"Unsupported devices type: {type(devices)}. Expected torch.device, int, list, tuple, or 'auto'.")
        return [torch.device("musa", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the number of MUSA devices when set to `auto`."""
        return torch_musa.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        """Checks if MUSA is available on the system."""
        return torch_musa.is_available()

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        """Register this accelerator with its name and description."""
        accelerator_registry.register(
            "musa",
            cls,
            description=cls.__name__,
            override=True,
        )
