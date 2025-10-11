# refer to: https://github.com/plae-tljg/GPT-SoVITS-Musa/blob/main/code_patches.zip

import logging
import os
import shutil
import subprocess
from typing import Any, Optional, Union

import torch
import torch_musa  # 添加MUSA支持
from typing_extensions import override

import pytorch_lightning as pl
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# from icecream import ic

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
        # ic(device)
        if device.type != "musa":
            raise MisconfigurationException(f"Device should be MUSA, got {device} instead.")
        # MUSA 也需要设置当前设备
        if device.index is None:
            device = torch.device(f"musa:0")
        # ic(device)
        torch.musa.set_device(device)


    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        # 修改设备检测逻辑以支持MUSA
        if device.type == "musa":
            return torch_musa.memory_stats(device)

    @override
    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""
        torch.musa.empty_cache()

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return torch.device("musa") if _musa_available else None

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> list[torch.device]:  # 改为Any，输出list[torch.device]
        """Gets parallel devices for the Accelerator."""
        # ic(devices)  # 保留调试

        if isinstance(devices, torch.device):
            # 新增：如果已经是单个device，返回列表包装
            return [devices]

        if devices is None or devices == "auto":
            # auto：使用所有可用设备
            num_devices = MUSAAccelerator.auto_device_count()
            devices = list(range(num_devices))

        elif isinstance(devices, int):
            # int：生成索引范围
            devices = list(range(devices))

        elif isinstance(devices, (list, tuple)):
            # 已列表：直接用
            pass

        else:
            # 其他：raise错误
            raise ValueError(f"Unsupported devices type: {type(devices)}. Expected torch.device, int, list, tuple, or 'auto'.")

        # 现在devices是索引列表，创建设备
        return [torch.device("musa", i) for i in devices]


    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the number of MUSA devices when set to `auto`."""
        # 直接使用我们的工具函数
        return torch_musa.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        """Checks if MUSA is available on the system."""
        # 直接使用我们的工具函数
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
        
# registry = _AcceleratorRegistry
# MUSAAccelerator.register_accelerators(registry)