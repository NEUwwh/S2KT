import glob
import os
import random
import re
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from loguru import logger
from omegaconf import DictConfig
from packaging import version

# Try importing MLU support
try:
    import torch_mlu  # noqa: F401
    _MLU_AVAILABLE = True
except ImportError:
    _MLU_AVAILABLE = False

# --- Enums & Constants ---

@unique
class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MLU = "mlu"

    @classmethod
    def from_str(cls, value: str) -> "DeviceType":
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown device type '{value}'! Must be one of {[e.value for e in cls]}.")

# --- Device Backend Strategies ---

class IDeviceBackend(ABC):
    """Abstract Strategy for Device Operations."""
    
    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        pass

    @abstractmethod
    def set_device(self, device_id: int) -> torch.device:
        pass

    @abstractmethod
    def to_device(self, src: Union[torch.Tensor, nn.Module], device_id: Optional[int] = None, non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
        pass

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        pass

class CPUBackend(IDeviceBackend):
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    def get_device_count(self) -> int:
        return 1

    def set_device(self, device_id: int) -> torch.device:
        return torch.device("cpu")

    def to_device(self, src: Union[torch.Tensor, nn.Module], device_id: Optional[int] = None, non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
        return src.cpu()

    def manual_seed(self, seed: int) -> None:
        pass  # torch.manual_seed handles CPU globally

class GPUBackend(IDeviceBackend):
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.GPU

    def get_device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_id: int) -> torch.device:
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")

    def to_device(self, src: Union[torch.Tensor, nn.Module], device_id: Optional[int] = None, non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
        kwargs = {'non_blocking': non_blocking} if isinstance(src, torch.Tensor) else {}
        if device_id is None:
            return src.cuda(**kwargs)
        return src.to(f'cuda:{device_id}', **kwargs)

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class MLUBackend(IDeviceBackend):
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.MLU

    def get_device_count(self) -> int:
        return torch.mlu.device_count() if _MLU_AVAILABLE and hasattr(torch, "mlu") else 0

    def set_device(self, device_id: int) -> torch.device:
        if not _MLU_AVAILABLE:
            raise RuntimeError("MLU support not available.")
        torch.mlu.set_device(device_id)
        return torch.device(f"mlu:{device_id}")

    def to_device(self, src: Union[torch.Tensor, nn.Module], device_id: Optional[int] = None, non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
        kwargs = {'non_blocking': non_blocking} if isinstance(src, torch.Tensor) else {}
        if device_id is None:
            return src.mlu(**kwargs)
        return src.to(f'mlu:{device_id}', **kwargs)

    def manual_seed(self, seed: int) -> None:
        if _MLU_AVAILABLE and hasattr(torch, "mlu"):
            torch.mlu.manual_seed(seed)
            torch.mlu.manual_seed_all(seed)

# --- Device Manager (Singleton) ---

class DeviceManager:
    """Singleton Manager to handle device context switching."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._backend: IDeviceBackend = GPUBackend() # Default
        self._current_device_type = DeviceType.GPU

    @classmethod
    def instance(cls) -> "DeviceManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_backend(self, device_type_str: str):
        dtype = DeviceType.from_str(device_type_str)
        with self._lock:
            if dtype == DeviceType.CPU:
                self._backend = CPUBackend()
            elif dtype == DeviceType.GPU:
                self._backend = GPUBackend()
            elif dtype == DeviceType.MLU:
                self._backend = MLUBackend()
            self._current_device_type = dtype
            logger.info(f"Set device type to `{dtype.value}`")

    @property
    def backend(self) -> IDeviceBackend:
        return self._backend

# --- Runtime Environment Manager ---

class RuntimeEnvironment:
    """Encapsulates environment configuration logic."""
    
    @staticmethod
    def is_distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    def get_rank() -> int:
        if RuntimeEnvironment.is_distributed():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def configure(env_cfg: DictConfig):
        """Sets up TF32, Seeds, and Determinism based on config."""
        # TF32 Configuration
        tf32_mode = env_cfg.get("TF32", False)
        RuntimeEnvironment._set_tf32(tf32_mode)

        # Seed & Determinism
        seed = env_cfg.get("SEED", 42)
        if seed is not None:
            RuntimeEnvironment._setup_determinacy(
                seed=seed + RuntimeEnvironment.get_rank(),
                deterministic=env_cfg.get("DETERMINISTIC", True),
                cudnn_enabled=env_cfg.get("CUDNN.ENABLED", True),
                cudnn_benchmark=env_cfg.get("CUDNN.BENCHMARK", False),
                cudnn_deterministic=env_cfg.get("CUDNN.DETERMINISTIC", True),
            )

    @staticmethod
    def _set_tf32(mode: bool):
        current_type = DeviceManager.instance().backend.device_type
        if current_type != DeviceType.GPU:
            if mode:
                raise RuntimeError(f"Device {current_type.value} does not support TF32.")
            return

        if version.parse(torch.__version__) >= version.parse("1.7.0"):
            torch.backends.cuda.matmul.allow_tf32 = mode
            torch.backends.cudnn.allow_tf32 = mode
            logger.info(f"{'Enable' if mode else 'Disable'} TF32 mode")
        elif mode:
             raise RuntimeError(f"Torch version {torch.__version__} does not support TF32.")

    @staticmethod
    def _setup_determinacy(seed: int, deterministic: bool, cudnn_enabled: bool, 
                           cudnn_benchmark: bool, cudnn_deterministic: bool):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        DeviceManager.instance().backend.manual_seed(seed)

        if deterministic:
            if DeviceManager.instance().backend.device_type == DeviceType.GPU:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            
            if version.parse(torch.__version__) >= version.parse("1.8.0"):
                torch.use_deterministic_algorithms(True)
            elif version.parse(torch.__version__) >= version.parse("1.7.0"):
                torch.set_deterministic(True)
            logger.info("Use deterministic algorithms.")

        if DeviceManager.instance().backend.device_type == DeviceType.GPU:
            if not cudnn_enabled:
                torch.backends.cudnn.enabled = False
                logger.info("Unset cudnn enabled.")
            if not cudnn_benchmark:
                torch.backends.cudnn.benchmark = False
                logger.info("Unset cudnn benchmark.")
            if cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                logger.info("Set cudnn deterministic.")


# --- Checkpoint System ---

class CheckpointRetentionPolicy:
    """Logic for determining which checkpoints to keep or delete."""
    
    def __init__(self, strategy: Union[int, List, Tuple, None]):
        self.strategy = strategy

    def should_remove_last(self, last_epoch: int) -> bool:
        if self.strategy is None:
            return True
        if isinstance(self.strategy, int):
            return last_epoch % self.strategy != 0
        if isinstance(self.strategy, (list, tuple)):
            return last_epoch not in self.strategy
        return False

class CheckpointManager:
    """Handles IO operations for checkpoints using Pathlib."""
    
    @staticmethod
    def save(ckpt: Dict, path: Union[str, Path]):
        path = Path(path)
        torch.save(ckpt, path)
        logger.info(f"Checkpoint {path} saved")

    @staticmethod
    def load(directory: Union[str, Path], path: Optional[str] = None) -> Dict:
        directory = Path(directory)
        if path is None:
            path = CheckpointManager._get_last_ckpt_path(directory)
        
        logger.info(f"Loading Checkpoint from '{path}'")
        # Use the global device manager to map location
        return torch.load(f=path, map_location=lambda s, l: DeviceManager.instance().backend.to_device(s))

    @staticmethod
    def _get_last_ckpt_path(directory: Path, pattern: str = r"^.+_[\d]*.pt$") -> Path:
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
            
        files = [f for f in directory.iterdir() if re.search(pattern, f.name)]
        files.sort(key=lambda x: x.name)
        
        if not files:
            raise FileNotFoundError(f"No checkpoint found in {directory} with pattern {pattern}")
        return files[-1]

    @staticmethod
    def backup_last(last_path: Union[str, Path], epoch: int, policy: CheckpointRetentionPolicy):
        last_path = Path(last_path)
        last_epoch = epoch - 1
        
        if policy.should_remove_last(last_epoch) and last_epoch != 0:
            if last_path.exists():
                backup_path = last_path.with_name(last_path.name + ".bak")
                last_path.rename(backup_path)

    @staticmethod
    def clear_backups(directory: Union[str, Path], pattern: str = "*.pt.bak"):
        directory = Path(directory)
        for backup_file in directory.glob(pattern):
            backup_file.unlink()

# --- Metrics System ---

class IMetric(ABC):
    @abstractmethod
    def reset(self): pass
    @abstractmethod
    def update(self, value: float, n: int = 1): pass
    @abstractmethod
    def value(self) -> float: pass

class AverageMetric(IMetric):
    """Thread-safe Average Meter."""
    def __init__(self):
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._sum = 0.0
            self._count = 0

    def update(self, value: float, n: int = 1):
        with self._lock:
            self._sum += value * n
            self._count += n

    def value(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count != 0 else 0.0

class MetricEntry:
    """DTO for Metric Registry."""
    def __init__(self, metric: IMetric, name: str, mtype: str, fmt: str, plt: bool, index: int):
        self.metric = metric
        self.name = name
        self.type = mtype
        self.fmt = fmt
        self.plt = plt
        self.index = index

class MetricRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        self._registry: Dict[str, MetricEntry] = {}
        self._lock = threading.Lock()

    def register(self, name: str, meter_type: str, fmt: str = "{:f}", plt: bool = False):
        with self._lock:
            if name in self._registry:
                raise ValueError(f"Meter '{name}' already existed.")
            
            entry = MetricEntry(
                metric=AverageMetric(),
                name=name,
                mtype=meter_type,
                fmt=fmt,
                plt=plt,
                index=len(self._registry)
            )
            self._registry[name] = entry

    def update(self, name: str, value: float, n: int = 1):
        # Optimistic read, only lock if necessary or inside the metric itself
        if name not in self._registry:
            raise ValueError(f"Meter '{name}' not found.")
        self._registry[name].metric.update(value, n)

    def reset(self):
        for entry in self._registry.values():
            entry.metric.reset()

    def get_avg(self, name: str) -> float:
        if name not in self._registry:
            raise ValueError(f"Meter '{name}' not found.")
        return self._registry[name].metric.value()

    def get_all_by_type(self, meter_type: str) -> Dict[str, float]:
        return {
            name: entry.metric.value() 
            for name, entry in self._registry.items() 
            if entry.type == meter_type
        }

    def print_meters(self, meter_type: str, use_logger: bool = True):
        sorted_entries = sorted(
            [e for e in self._registry.values() if e.type == meter_type],
            key=lambda x: x.index
        )
        
        print_list = [
            f"{e.name}: {e.fmt.format(e.metric.value())}" 
            for e in sorted_entries
        ]
        
        msg = f"Result <{meter_type}>: [{', '.join(print_list)}]"
        if use_logger:
            logger.info(msg)
        else:
            print(msg)
    
    def plt_meters(self, meter_type: str):
        pass

# --- Legacy Functional Interface (Proxy Pattern) ---
# To maintain backward compatibility with the original function calls

def get_device_type() -> str:
    return DeviceManager.instance().backend.device_type.value

def set_device_type(device_type: str):
    DeviceManager.instance().set_backend(device_type)

def get_device_count() -> int:
    return DeviceManager.instance().backend.get_device_count()

def set_device(device_id: int):
    return DeviceManager.instance().backend.set_device(device_id)

def to_device(src: Union[torch.Tensor, nn.Module], device_id: int = None, non_blocking: bool = False) -> Union[torch.Tensor, nn.Module]:
    return DeviceManager.instance().backend.to_device(src, device_id, non_blocking)

def set_device_manual_seed(seed: int):
    DeviceManager.instance().backend.manual_seed(seed)

def is_distributed() -> bool:
    return RuntimeEnvironment.is_distributed()

def get_rank() -> int:
    return RuntimeEnvironment.get_rank()

def set_tf32_mode(tf32_mode: bool):
    RuntimeEnvironment._set_tf32(tf32_mode)

def setup_determinacy(seed: int, deterministic: bool = False, cudnn_enabled: bool = True, cudnn_benchmark: bool = False, cudnn_deterministic: bool = True):
    RuntimeEnvironment._setup_determinacy(seed, deterministic, cudnn_enabled, cudnn_benchmark, cudnn_deterministic)

def set_env(env_cfg: DictConfig):
    RuntimeEnvironment.configure(env_cfg)

def save_ckpt(ckpt: Dict, ckpt_path: str):
    CheckpointManager.save(ckpt, ckpt_path)

def load_ckpt(ckpt_save_dir: str, ckpt_path: str = None) -> Dict:
    return CheckpointManager.load(ckpt_save_dir, ckpt_path)

def get_last_ckpt_path(ckpt_save_dir: str, name_pattern: str = r"^.+_[\d]*.pt$") -> str:
    return str(CheckpointManager._get_last_ckpt_path(Path(ckpt_save_dir), name_pattern))

def need_to_remove_last_ckpt(last_epoch: int, ckpt_save_strategy: Union[int, List, Tuple]) -> bool:
    return CheckpointRetentionPolicy(ckpt_save_strategy).should_remove_last(last_epoch)

def backup_last_ckpt(last_ckpt_path: str, epoch: int, ckpt_save_strategy: Union[int, List, Tuple]):
    CheckpointManager.backup_last(last_ckpt_path, epoch, CheckpointRetentionPolicy(ckpt_save_strategy))

def clear_ckpt(ckpt_save_dir: str, name_pattern: str = "*.pt.bak"):
    CheckpointManager.clear_backups(ckpt_save_dir, name_pattern)

# Alias for compatibility
AvgMeter = AverageMetric
MeterPool = MetricRegistry

__all__ = [
    "get_rank", "is_distributed", "set_tf32_mode", "setup_determinacy", "set_env",
    "get_device_type", "set_device_type", "get_device_count", "set_device", "to_device", "set_device_manual_seed",
    "save_ckpt", "load_ckpt", "get_last_ckpt_path", "need_to_remove_last_ckpt", "backup_last_ckpt", "clear_ckpt",
    "AvgMeter", "MeterPool"
]