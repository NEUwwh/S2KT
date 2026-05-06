import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime  # <--- 之前缺少了这一行
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Type

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Assuming these exist in the user's codebase context
from scripts.data_preparation import data_preparation_pipeline
from .utils import (
    MeterPool,
    backup_last_ckpt,
    clear_ckpt,
    get_device_count,
    get_rank,
    load_ckpt,
    save_ckpt,
    set_env,
    to_device,
)
from .datasets import BaseKnowledgeTracingDataset
from .metrics import masked_acc, masked_auc

# --- Core Infrastructure & Enums ---

class RunPhase(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

@dataclass
class RunContext:
    """Encapsulates the dynamic state of the training/evaluation process."""
    epoch: int = 0
    num_epochs: int = 0
    global_step: int = 0
    cfg: DictConfig = field(default_factory=lambda: DictConfig({}))
    model_name: str = ""
    start_time: float = field(default_factory=time.time)

# --- Registry System ---

class ComponentRegistry:
    """Central registry for custom optimizers and schedulers."""
    _optimizers: Dict[str, Type[optim.Optimizer]] = {}
    _schedulers: Dict[str, Type[Any]] = {}

    @classmethod
    def register_optimizer(cls, name: str):
        def decorator(optim_cls):
            cls._optimizers[name] = optim_cls
            return optim_cls
        return decorator

    @classmethod
    def register_scheduler(cls, name: str):
        def decorator(scheduler_cls):
            cls._schedulers[name] = scheduler_cls
            return scheduler_cls
        return decorator

    @classmethod
    def get_optimizer(cls, name: str) -> Optional[Type[optim.Optimizer]]:
        return cls._optimizers.get(name)

    @classmethod
    def get_scheduler(cls, name: str) -> Optional[Type[Any]]:
        return cls._schedulers.get(name)

# Placeholder for backward compatibility with original code's kt_optimizer/kt_lr_scheduler
kt_optimizer = type("LegacyOptimizerRegistry", (), {"__getattr__": lambda s, k: ComponentRegistry.get_optimizer(k)})()
kt_lr_scheduler = type("LegacySchedulerRegistry", (), {"__getattr__": lambda s, k: ComponentRegistry.get_scheduler(k)})()


# --- Component Factories ---

class OptimizerFactory:
    @staticmethod
    def create(optimizer_cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
        if isinstance(optimizer_cfg.TYPE, type):
            optim_type = optimizer_cfg.TYPE
        elif hasattr(optim, optimizer_cfg.TYPE):
            optim_type = getattr(optim, optimizer_cfg.TYPE)
        else:
            optim_type = ComponentRegistry.get_optimizer(optimizer_cfg.TYPE)
            if optim_type is None:
                 # Fallback to checking the legacy/module level if needed, or raise error
                 raise ValueError(f"Optimizer {optimizer_cfg.TYPE} not found.")

        optimizer_param = optimizer_cfg.get("PARAM", {})
        logger.debug(f"Instantiating optimizer {optim_type.__name__} with params: {optimizer_param}")
        return optim_type(model.parameters(), **optimizer_param)

class SchedulerFactory:
    @staticmethod
    def create(lr_scheduler_cfg: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        if isinstance(lr_scheduler_cfg.TYPE, type):
            scheduler_type = lr_scheduler_cfg.TYPE
        elif hasattr(lr_scheduler, lr_scheduler_cfg.TYPE):
            scheduler_type = getattr(lr_scheduler, lr_scheduler_cfg.TYPE)
        else:
            scheduler_type = ComponentRegistry.get_scheduler(lr_scheduler_cfg.TYPE)
            if scheduler_type is None:
                raise ValueError(f"Scheduler {lr_scheduler_cfg.TYPE} not found.")

        scheduler_param = lr_scheduler_cfg.get("PARAM").copy()
        scheduler_param["optimizer"] = optimizer
        return scheduler_type(**scheduler_param)


# --- Sub-Managers ---

class EnvironmentManager:
    """Handles environment setup and device management."""
    def __init__(self, env_cfg: DictConfig):
        self.env_cfg = env_cfg

    def setup(self):
        logger.info("=" * 30 + " SETTING ENVIRONMENT " + "=" * 30)
        set_env(env_cfg=self.env_cfg)

    def to_device(self, model: nn.Module) -> nn.Module:
        return to_device(model)

class CheckpointManager:
    """Handles saving, loading, and managing model checkpoints."""
    def __init__(self, cfg: DictConfig, model_class_name: str, model_name_dataset: str):
        self.cfg = cfg
        self.model_class_name = model_class_name
        self.save_dir = self._generate_save_dir(model_name_dataset)
        self.save_strategy = cfg.TRAIN_CFG.get("CKPT_SAVE_STRATEGY", None)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _generate_save_dir(self, dataset_name: str) -> str:
        md5 = hashlib.md5(OmegaConf.to_yaml(self.cfg).encode()).hexdigest()
        return os.path.join(self.cfg.TRAIN_CFG.get("CKPT_SAVE_DIR", "checkpoints"),
                            self.model_class_name,
                            dataset_name,
                            md5)

    def get_ckpt_path(self, epoch: int, num_epochs: int) -> str:
        epoch_str = str(epoch).zfill(len(str(num_epochs)))
        return os.path.join(self.save_dir, f"{self.model_class_name}_{epoch_str}.pt")

    def save(self, context: RunContext, model: nn.Module, optimizer: optim.Optimizer, best_metrics: Dict):
        ckpt = {
            "epoch": context.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metrics": best_metrics
        }
        
        last_ckpt_path = self.get_ckpt_path(context.epoch - 1, context.num_epochs)
        backup_last_ckpt(last_ckpt_path=last_ckpt_path, epoch=context.epoch, ckpt_save_strategy=self.save_strategy)
        
        current_path = self.get_ckpt_path(context.epoch, context.num_epochs)
        save_ckpt(ckpt=ckpt, ckpt_path=current_path)

        if context.epoch % 10 == 0 or context.epoch == context.num_epochs:
            clear_ckpt(ckpt_save_dir=self.save_dir)

    def save_best(self, context: RunContext, model: nn.Module, optimizer: optim.Optimizer, 
                  best_metrics: Dict, metric_name: str):
        ckpt = {
            "epoch": context.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metrics": best_metrics
        }
        ckpt_path = os.path.join(
            self.save_dir,
            f"{self.model_class_name}_best_{metric_name.replace('/', '_')}.pt"
        )
        save_ckpt(ckpt=ckpt, ckpt_path=ckpt_path)

    def load(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, 
             ckpt_path: Optional[str] = None, strict: bool = True) -> Dict:
        """Loads checkpoint and returns the loaded dictionary."""
        try:
            ckpt = load_ckpt(ckpt_save_dir=self.save_dir, ckpt_path=ckpt_path)
            model.load_state_dict(ckpt.get("model_state_dict"), strict=strict)
            if optimizer and ckpt.get("optimizer_state_dict"):
                optimizer.load_state_dict(ckpt.get("optimizer_state_dict"))
            return ckpt
        except (IndexError, OSError, KeyError) as e:
            if ckpt_path: # If explicit path failed, raise error
                raise OSError(f"Error opening ckpt file: {ckpt_path}") from e
            return {} # If implicit load failed, return empty (start fresh)

class MetricComputer:
    """Encapsulates the logic for computing metrics dynamically based on signature."""
    @staticmethod
    def compute(metric_func: Union[Callable, functools.partial], args: Dict[str, torch.Tensor]) -> torch.Tensor:
        func = metric_func.func if isinstance(metric_func, functools.partial) else metric_func
        sig = inspect.signature(func)
        required_keys = set(sig.parameters.keys())
        
        filtered_args = {}
        for k, v in args.items():
            if k in required_keys:
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    filtered_args[k] = v.detach()
                else:
                    filtered_args[k] = v
        
        return metric_func(**filtered_args)

# --- Main Abstract Runner ---

class BaseEpochRunner(ABC):
    """
    Abstract Base Runner implementing the Template Method pattern for the training loop.
    Decomposes responsibilities into managers.
    """

    def __init__(self, cfg: DictConfig):
        self.context = RunContext(cfg=cfg)
        
        # Managers
        self.env_manager = EnvironmentManager(cfg.get("ENV_CFG", {}))
        self.env_manager.setup()
        
        # Model Construction
        self.model = self._build_model_wrapped(cfg.MODEL_CFG)
        
        # Optimization (Late Init)
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Data (Late Init)
        self.train_loader: Optional[DataLoader] = None
        self.valid_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        # Metrics & State
        self.meter_pool = MeterPool()
        self.best_metrics: Dict[str, float] = {}
        self.target_metric: Optional[str] = None
        
        # Control Flow Params
        self.valid_interval = 1
        self.test_interval = 1
        
        # Checkpointing (Late Init to allow subclasses to define model name logic)
        self.ckpt_manager: Optional[CheckpointManager] = None
        self.early_stop_patience: Optional[int] = None
        self.current_patience: Optional[int] = None

    def _build_model_wrapped(self, model_cfg: DictConfig) -> nn.Module:
        """Internal wrapper to build and move model to device."""
        model = self.build_model(model_cfg)
        return self.env_manager.to_device(model)

    def build_model(self, model_cfg: DictConfig) -> nn.Module:
        """Reflection-based model builder."""
        module = importlib.import_module(model_cfg.ROOT_PATH)
        model_class = getattr(module, model_cfg.MODEL_NAME)
        # Dynamic argument injection
        valid_params = {k: model_cfg.MODEL_PARAMS[k] for k in model_cfg.MODEL_PARAMS 
                        if k in model_class.__init__.__code__.co_varnames}
        return model_class(**valid_params)

    # --- Abstract Data Methods ---
    @abstractmethod
    def build_train_dataset(self, data_cfg: DictConfig) -> Dataset: raise NotImplementedError
    @abstractmethod
    def build_test_dataset(self, data_cfg: DictConfig) -> Dataset: raise NotImplementedError
    def build_valid_dataset(self, data_cfg: DictConfig) -> Dataset: pass

    # --- Data Loader Builders ---
    def _build_loader(self, dataset: Dataset, cfg_section: DictConfig, name: str) -> DataLoader:
        logger.info(f"Building {name} data loader.")
        return DataLoader(dataset=dataset,
                          batch_size=cfg_section.get("BATCH_SIZE", 32),
                          shuffle=cfg_section.get("SHUFFLE", True),
                          pin_memory=cfg_section.get("PIN_MEMORY", False))

    # --- Core Lifecycle ---
    
    # --- Core Lifecycle ---
    
    def init_train(self, cfg: DictConfig):
        logger.info("=" * 30 + " INITIALIZING TRAINING " + "=" * 30)
        self.optimizer = OptimizerFactory.create(cfg.TRAIN_CFG.get("OPTIMIZER_CFG", {}), self.model)
        
        # [FIXED]: 指向 cfg.DATA_CFG.TRAIN_SET 而不是 cfg.TRAIN_SET
        self.train_loader = self._build_loader(
            self.build_train_dataset(cfg.DATA_CFG), 
            cfg.DATA_CFG.TRAIN_SET, 
            "train"
        )
        
        # Context Setup
        self.context.num_epochs = cfg.TRAIN_CFG.NUM_EPOCHS
        self.context.start_epoch = 0
        
        # Early Stopping
        self.early_stop_patience = cfg.TRAIN_CFG.get("EARLY_STOPPING_PATIENCE", self.context.num_epochs)
        self.current_patience = self.early_stop_patience
        assert self.early_stop_patience > 0, "Patience must be positive."

        # Checkpoint Manager
        self.ckpt_manager = CheckpointManager(cfg, self.model.__class__.__name__, cfg.DATA_CFG.DATASET_NAME)

        # Logging
        self.meter_pool.register("train/time", "train", "{:.2f} (s)", plt=False)
        self._setup_file_logger()

        # Resume
        self.load_model_resume()

        # Optional Phases
        if cfg.get("VALID_CFG"): self.init_valid(cfg)
        if cfg.get("TEST_CFG"): self.init_test(cfg)

    def init_valid(self, cfg: DictConfig):
        logger.info("=" * 30 + " INITIALIZING VALIDATION " + "=" * 30)
        self.valid_loader = self._build_loader(
            self.build_valid_dataset(cfg.DATA_CFG), 
            cfg.DATA_CFG.VALID_SET, 
            "valid"
        )
        self.valid_interval = cfg.VALID_CFG.get("VALID_INTERVAL", 1)
        self.meter_pool.register("valid/time", "valid", "{:.2f} (s)", plt=False)

    def init_test(self, cfg: DictConfig):
        logger.info("=" * 30 + " INITIALIZING TESTING " + "=" * 30)
        # [FIXED]: 指向 cfg.DATA_CFG.TEST_SET
        self.test_loader = self._build_loader(
            self.build_test_dataset(cfg.DATA_CFG), 
            cfg.DATA_CFG.TEST_SET, 
            "test"
        )
        self.test_interval = cfg.VALID_CFG.get("TEST_INTERVAL", 1)
        self.meter_pool.register("test/time", "test", "{:.2f} (s)", plt=False)

    def _setup_file_logger(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.ckpt_manager.save_dir, f"training_log_{timestamp}.log")
        logger.add(log_file, level="INFO", encoding="utf-8")

    def train(self, cfg: DictConfig):
        self.init_train(cfg)
        self.on_train_start()

        for epoch_idx in range(self.context.start_epoch, self.context.num_epochs):
            self.context.epoch = epoch_idx + 1
            if self._check_early_stopping(): break

            self.on_epoch_start(self.context.epoch)
            t0 = time.time()

            self.model.train()
            self._run_epoch_loop()

            self.meter_pool.update("train/time", time.time() - t0)
            self.on_epoch_end(self.context.epoch)

        logger.info(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.on_train_end(cfg, self.context.epoch)

    def _run_epoch_loop(self):
        """Standard training loop with TQDM wrapper."""
        loader = self._wrap_tqdm(self.train_loader)
        for _, data in enumerate(loader):
            loss = self.train_iter(data)
            if loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @abstractmethod
    def train_iter(self, data: Dict[str, torch.Tensor]) -> torch.Tensor: pass
    @abstractmethod
    def validate(self): pass
    @abstractmethod
    def test(self, save_results: bool = True, save_metrics: bool = True): pass

    # --- Pipeline Triggers ---
    
    @torch.no_grad()
    def validate_pipeline(self, cfg: DictConfig = None, train_epoch: Optional[int] = None):
        if train_epoch is None: self.init_valid(cfg)
        self.on_valid_start(train_epoch)
        t0 = time.time()
        self.model.eval()
        self.validate()
        self.meter_pool.update("valid/time", time.time() - t0)
        self.on_valid_end(train_epoch)

    @torch.no_grad()
    def test_pipeline(self, cfg: DictConfig = None, train_epoch: Optional[int] = None):
        if train_epoch is None and cfg is not None: self.init_test(cfg)
        self.on_test_start(train_epoch)
        t0 = time.time()
        self.model.eval()
        self.test()
        self.meter_pool.update("test/time", time.time() - t0)
        self.on_test_end(train_epoch)

    # --- Hooks & Helpers ---

    def on_train_start(self): logger.info("=" * 30 + " STARTING TRAINING " + "=" * 30)
    
    def on_train_end(self, cfg: DictConfig, train_epoch: int):
        if cfg.get("TEST_CFG"):
            # Load best model for testing
            best_model_name = f"{self.model.__class__.__name__}_best_valid_{self.target_metric.replace('/', '_')}.pt"
            best_path = os.path.join(self.ckpt_manager.save_dir, best_model_name)
            logger.info("Evaluating best model on test set.")
            self.load_model(best_path, strict=True)
            self.test_pipeline(cfg, train_epoch)
        
        # Backup config
        OmegaConf.save(config=cfg, f=os.path.join(self.ckpt_manager.save_dir, "config.yaml"))

    def on_epoch_start(self, epoch: int): logger.info(f"Epoch {epoch} / {self.context.num_epochs}")

    def on_epoch_end(self, epoch: int):
        self.meter_pool.print_meters("train")
        if self.valid_loader and epoch % self.valid_interval == 0:
            self.validate_pipeline(train_epoch=epoch)
        if self.test_loader and epoch % self.test_interval == 0:
            self.test_pipeline(train_epoch=epoch)
        
        self.ckpt_manager.save(self.context, self.model, self.optimizer, self.best_metrics)
        self.meter_pool.reset()

    def on_valid_start(self, epoch: Optional[int]): logger.info("Start validating.")
    def on_valid_end(self, epoch: Optional[int]): 
        self.meter_pool.print_meters("valid")
        # Logic for saving best model
        if self.metric_best != "min": # Greater is best
            greater_best = True
        else:
            greater_best = False
            
        if epoch is not None:
            self._save_best_model_logic(epoch, "valid/" + self.target_metric, greater_best)

    def on_test_start(self, epoch: Optional[int]): logger.info("Start testing.")
    def on_test_end(self, epoch: Optional[int]): self.meter_pool.print_meters("test")

    def _save_best_model_logic(self, epoch: int, metric_name: str, greater_best: bool):
        current_val = self.meter_pool.get_avg(metric_name)
        best_val = self.best_metrics.get(metric_name)

        is_improvement = False
        if best_val is None:
            is_improvement = True
        elif greater_best and current_val > best_val:
            is_improvement = True
        elif not greater_best and current_val < best_val:
            is_improvement = True

        if is_improvement:
            self.best_metrics[metric_name] = current_val
            self.ckpt_manager.save_best(self.context, self.model, self.optimizer, self.best_metrics, metric_name)
            self.current_patience = self.early_stop_patience
        else:
            if self.early_stop_patience: self.current_patience -= 1

    def _check_early_stopping(self) -> bool:
        if self.current_patience <= 0:
            logger.info("Early stopping triggered.")
            return True
        return False

    def load_model_resume(self, strict: bool = True):
        ckpt = self.ckpt_manager.load(self.model, self.optimizer, strict=strict)
        if ckpt:
            self.context.start_epoch = ckpt.get("epoch", 0)
            self.best_metrics = ckpt.get("best_metrics", {})
            logger.info("Resumed training from checkpoint.")

    def load_model(self, ckpt_path: str, strict: bool = True):
        self.ckpt_manager.load(self.model, ckpt_path=ckpt_path, strict=strict)

    # Delegation methods exposed to subclasses
    def register_epoch_meter(self, name: str, mtype: str, fmt: str = "{:f}", plt: bool = False):
        self.meter_pool.register(name, mtype, fmt, plt)
    
    def update_epoch_meter(self, name: str, val: float, n: int = 1):
        self.meter_pool.update(name, val, n)

    def _wrap_tqdm(self, loader: DataLoader) -> Union[DataLoader, tqdm]:
        local_rank = get_rank() % get_device_count() if get_device_count() != 0 else 0
        return tqdm(loader, file=sys.stdout) if local_rank == 0 else loader


# --- Concrete Implementation ---

class BaseKnowledgeTracingRunner(BaseEpochRunner):
    """
    Concrete implementation of the Knowledge Tracing Runner.
    Implements specific dataset building, metric registration, and iteration logic.
    """
    
    DEFAULT_METRICS = {
        "AUC": masked_auc,
        "ACC": masked_acc,
    }

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model_name = cfg.MODEL_CFG.MODEL_NAME
        self._configure_metrics(cfg.METRICS_CFG)
        self._prepare_data(cfg.DATA_CFG)

    def _prepare_data(self, data_cfg: DictConfig):
        data_preparation_pipeline(
            data_cfg=data_cfg,
            dataset_name=data_cfg.DATASET_NAME,
            model_name=self.model_name
        )

    def _configure_metrics(self, metrics_cfg: DictConfig):
        metric_types = metrics_cfg.get("TYPE", list(self.DEFAULT_METRICS.keys()))
        unknown = set(metric_types) - set(self.DEFAULT_METRICS.keys())
        if unknown: raise ValueError(f"Unknown metrics: {unknown}")

        self.active_metrics = {k: self.DEFAULT_METRICS[k] for k in metric_types}
        self.target_metric = metrics_cfg.get("TARGET", "loss")
        self.metric_best = metrics_cfg.get("BEST", "max")
        
        if self.target_metric != "loss" and self.target_metric not in self.active_metrics:
             raise ValueError(f"Target metric {self.target_metric} not enabled.")
        if self.metric_best not in ["max", "min"]:
             raise ValueError("Invalid metric_best criteria.")

    # --- Dataset Implementation ---
    def build_train_dataset(self, data_cfg: DictConfig) -> Dataset:
        return BaseKnowledgeTracingDataset(data_cfg, self.model_name, "train")

    def build_valid_dataset(self, data_cfg: DictConfig) -> Dataset:
        return BaseKnowledgeTracingDataset(data_cfg, self.model_name, "valid")

    def build_test_dataset(self, data_cfg: DictConfig) -> Dataset:
        return BaseKnowledgeTracingDataset(data_cfg, self.model_name, "test")

    # --- Initialization Overrides (to register specific metrics) ---
    def init_train(self, cfg: DictConfig):
        super().init_train(cfg)
        self._register_phase_metrics("train")

    def init_valid(self, cfg: DictConfig):
        super().init_valid(cfg)
        self._register_phase_metrics("valid")

    def init_test(self, cfg: DictConfig):
        super().init_test(cfg)
        self._register_phase_metrics("test")

    def _register_phase_metrics(self, phase: str):
        self.register_epoch_meter(f"{phase}/loss", phase, "{:.4f}")
        for key in self.active_metrics:
            self.register_epoch_meter(f"{phase}/{key}", phase, "{:.4f}")

    # --- Core Logic ---
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(feed_dict=data)

    def train_iter(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward(data)
        loss = self.model.masked_loss(out_dict=out)
        self.update_epoch_meter("train/loss", loss.item())
        
        for name, func in self.active_metrics.items():
            val = MetricComputer.compute(func, out)
            self.update_epoch_meter(f"train/{name}", val.item())
        return loss

    @torch.no_grad()
    def validate(self):
        self._run_evaluation_loop(self.valid_loader, "valid")

    @torch.no_grad()
    def test(self, save_results: bool = True, save_metrics: bool = True):
        results, metrics = self._run_evaluation_loop(self.test_loader, "test")
        
        if save_results:
            np.savez(os.path.join(self.ckpt_manager.save_dir, "test_results.npz"),
                     **{k: v.cpu().numpy() for k, v in results.items()})
        if save_metrics:
            with open(os.path.join(self.ckpt_manager.save_dir, "test_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

    def _run_evaluation_loop(self, loader: DataLoader, phase: str) -> Tuple[Dict, Dict]:
        """Unified evaluation loop for validation and testing."""
        preds, targets, masks = [], [], []
        
        for _, data in enumerate(self._wrap_tqdm(loader)):
            out = self.forward(data)
            loss = self.model.masked_loss(out_dict=out)
            self.update_epoch_meter(f"{phase}/loss", loss.item())

            preds.append(out.get("prediction"))
            targets.append(out.get("target"))
            masks.append(out.get("mask"))
        
        # Concat
        results = {
            "prediction": torch.cat(preds, dim=0),
            "target": torch.cat(targets, dim=0),
            "mask": torch.cat(masks, dim=0)
        }

        # Compute Metrics
        metrics_computed = {}
        for name, func in self.active_metrics.items():
            val = MetricComputer.compute(func, results)
            self.update_epoch_meter(f"{phase}/{name}", val.item())
            metrics_computed[name] = val.item()

        return results, metrics_computed


# Exposed factory wrappers to maintain import compatibility if needed, 
# though usage within class is now updated.
def build_optimizer(optimizer_cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    return OptimizerFactory.create(optimizer_cfg, model)

def build_lr_scheduler(lr_scheduler_cfg: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
    return SchedulerFactory.create(lr_scheduler_cfg, optimizer)

__all__ = ["BaseEpochRunner", "BaseKnowledgeTracingRunner", "build_optimizer", "build_lr_scheduler"]