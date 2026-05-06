import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar

import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import Dataset

# --- Type Definitions ---
T = TypeVar("T")
TensorDict = Dict[str, torch.Tensor]

# --- Enums & Constants ---
class DatasetMode(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    @classmethod
    def from_str(cls, value: str) -> "DatasetMode":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid mode: {value}. Supported modes: {[m.value for m in cls]}")

# --- Abstract Base Classes ---

class IDataProcessor(ABC):
    """Interface for data processing strategies."""
    
    @abstractmethod
    def process(self, series: pd.Series) -> torch.Tensor:
        """Process a pandas Series into a Tensor."""
        pass

class BaseDataset(Dataset):
    """Abstract base dataset enforcing implementation of core methods."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples."""
        raise NotImplementedError("`__len__` method must be implemented.")

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Retrieve a sample."""
        raise NotImplementedError("`__getitem__` method must be implemented.")

    @abstractmethod
    def _load_data(self) -> Any:
        """Internal method to load data."""
        raise NotImplementedError("`_load_data` method must be implemented.")


# --- Helper Classes / Components ---

class DeviceManager:
    """Singleton-like helper to determine computation device."""
    
    _instance = None
    
    @classmethod
    def get_device(cls) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

class SequenceParser:
    """Utilities for parsing stringified sequences from CSV."""
    
    @staticmethod
    def parse_float_seq(x: Any) -> List[float]:
        return [float(i) for i in str(x).split(",")]

    @staticmethod
    def parse_bool_mask_seq(x: Any, mask_val: int) -> List[bool]:
        return [int(i) != mask_val for i in str(x).split(",")]

# --- Column Processing Strategies ---

class BaseColumnProcessor(IDataProcessor):
    def __init__(self, device: torch.device):
        self.device = device

class FloatSequenceProcessor(BaseColumnProcessor):
    def process(self, series: pd.Series) -> torch.Tensor:
        return torch.tensor(
            series.apply(SequenceParser.parse_float_seq).tolist(),
            dtype=torch.float,
            device=self.device
        )

class BoolMaskProcessor(BaseColumnProcessor):
    def __init__(self, device: torch.device, mask_val: int = -1):
        super().__init__(device)
        self.mask_val = mask_val

    def process(self, series: pd.Series) -> torch.Tensor:
        return torch.tensor(
            series.apply(lambda x: SequenceParser.parse_bool_mask_seq(x, self.mask_val)).tolist(),
            dtype=torch.bool,
            device=self.device
        )

class LongSequenceProcessor(BaseColumnProcessor):
    def process(self, series: pd.Series) -> torch.Tensor:
        return torch.tensor(
            series.apply(SequenceParser.parse_float_seq).tolist(), # Original code parsed as float then cast to long implicitly/explicitly logic
            dtype=torch.long,
            device=self.device
        )

class DataFrameTensorizer:
    """Orchestrator for converting DataFrame columns to Tensors based on rules."""
    
    def __init__(self, mask_val: int = -1):
        self.device = DeviceManager.get_device()
        self.mask_val = mask_val
        self._strategy_map: Dict[str, IDataProcessor] = {
            "response": FloatSequenceProcessor(self.device),
            "select_mask": BoolMaskProcessor(self.device, mask_val),
        }
        self._default_processor = LongSequenceProcessor(self.device)

    def convert(self, df: DataFrame) -> TensorDict:
        data: TensorDict = {}
        for col_name in df.columns:
            processor = self._strategy_map.get(col_name, self._default_processor)
            data[col_name] = processor.process(df[col_name])
        return data


# --- Main Implementation ---

class BaseKnowledgeTracingDataset(BaseDataset):
    def __init__(self, data_cfg: DictConfig, model_name: str, mode: str):
        """
        A robust implementation of the knowledge tracing dataset.
        
        Uses Strategy pattern for data processing and Pathlib for file handling.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.mode = DatasetMode.from_str(mode)
        
        # Path construction using Pathlib for better OS compatibility
        self.root_path = Path(self.data_cfg.ROOT_PATH)
        self.dataset_dir = self.root_path / self.data_cfg.DATASET_NAME / "model_data" / model_name
        
        self._setup_paths()
        
        logger.info(f"Start preprocessing {self.pkl_file_path} ...")
        self.tensorizer = DataFrameTensorizer()
        self.data: TensorDict = self._load_data()

    def _setup_paths(self) -> None:
        """Configures file paths based on the operational mode."""
        files_cfg = self.data_cfg.FILES
        
        if self.mode == DatasetMode.TRAIN:
            self.log_file_path = self.dataset_dir / files_cfg.TRAIN_DATA.NAME
            suffix = files_cfg.TRAIN_SET.NAME
        elif self.mode == DatasetMode.VALID:
            self.log_file_path = self.dataset_dir / files_cfg.VALID_DATA.NAME
            suffix = files_cfg.VALID_SET.NAME
        elif self.mode == DatasetMode.TEST:
            self.log_file_path = self.dataset_dir / files_cfg.TEST_DATA.NAME
            suffix = files_cfg.TEST_SET.NAME
        else:
            # Should be unreachable due to Enum validation
            raise ValueError("Unreachable code path for mode setup.")
            
        self.pkl_file_path = self.dataset_dir / f"{self.__class__.__name__}_{suffix}"

    def __len__(self) -> int:
        """
        Return the number of sequences in the dataset.
        Assumes all columns have the same length.
        """
        if not self.data:
            return 0
        # Peek at the first value safely
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    def __getitem__(self, index: int) -> TensorDict:
        """
        Return the sample at the given index using dictionary comprehension.
        """
        return {key: tensor[index] for key, tensor in self.data.items()}

    def _load_data(self) -> TensorDict:
        """
        Orchestrates the loading of raw logs, processing, and caching.
        """
        raw_df = self._read_log_file()
        processed_data = self._process_dataframe(raw_df)
        self._cache_data(processed_data)
        return processed_data

    def _read_log_file(self) -> DataFrame:
        """Reads the CSV log file safely."""
        path_str = str(self.log_file_path)
        if not self.log_file_path.exists():
            raise FileNotFoundError(f"Log file not found at: {path_str}")
        
        try:
            return pd.read_csv(path_str, dtype=str)
        except ValueError as e:
            raise ValueError(f"Error parsing CSV file {path_str}: {str(e)}") from e

    def _process_dataframe(self, df: DataFrame, mask_val: int = -1) -> TensorDict:
        """
        Delegates processing to the DataFrameTensorizer.
        """
        # Note: mask_val passed here is for API compatibility, 
        # though Tensorizer is initialized in __init__
        return self.tensorizer.convert(df)

    def _cache_data(self, data: TensorDict) -> None:
        """Saves the processed data to a pickle file."""
        try:
            pd.to_pickle(data, str(self.pkl_file_path))
        except Exception as e:
            logger.warning(f"Failed to cache processed data to {self.pkl_file_path}: {e}")


__all__ = ["BaseDataset", "BaseKnowledgeTracingDataset"]