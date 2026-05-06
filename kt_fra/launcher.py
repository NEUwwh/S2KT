import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type, NoReturn

from omegaconf import DictConfig

from kt_fra.runner import BaseKnowledgeTracingRunner

# --- Configuration & Logging ---
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the model checkpoint cannot be located."""
    pass

class RunnerInitializationError(RuntimeError):
    """Raised when the runner fails to initialize."""
    pass

# --- Helper Components ---

class CheckpointResolver:
    """
    Responsible for resolving the correct path to the model checkpoint.
    Implements the fallback logic: Explicit Path -> Auto Discovery -> Error.
    """
    def __init__(self, runner: BaseKnowledgeTracingRunner):
        self._runner = runner

    def resolve(self, requested_path: Optional[str]) -> Path:
        """
        Resolves the checkpoint path.
        
        Args:
            requested_path (str, optional): The path provided by the user.

        Returns:
            Path: The validated path to the checkpoint file.

        Raises:
            CheckpointNotFoundError: If no valid checkpoint is found.
        """
        # Logic: If path is None OR path does not exist, try auto-discovery
        if not requested_path or not Path(requested_path).exists():
            if requested_path and not Path(requested_path).exists():
                logger.warning(f"Provided checkpoint path does not exist: {requested_path}. Attempting auto-discovery.")
            
            return self._auto_discover()
        
        logger.info(f"Using explicitly provided checkpoint: {requested_path}")
        return Path(requested_path)

    def _auto_discover(self) -> Path:
        """Constructs the default checkpoint path based on runner configuration."""
        save_dir = Path(self._runner.ckpt_save_dir)
        # Handle metric name sanitization (replacing slash with underscore)
        metric_suffix = self._runner.target_metric.replace('/', '_')
        model_name = self._runner.model.__class__.__name__
        
        filename = f"{model_name}_best_valid_{metric_suffix}.pt"
        auto_path = save_dir / filename

        if not auto_path.exists():
            raise CheckpointNotFoundError(
                f"Could not locate checkpoint automatically at {auto_path}. "
                "Please ensure training has completed or provide a valid path."
            )
        
        logger.info(f"Auto-discovered checkpoint: {auto_path}")
        return auto_path


class RunnerFactory:
    """Factory to handle the instantiation of the Knowledge Tracing Runner."""
    
    @staticmethod
    def create(cfg: DictConfig) -> BaseKnowledgeTracingRunner:
        try:
            logger.debug("Instantiating BaseKnowledgeTracingRunner...")
            return BaseKnowledgeTracingRunner(cfg=cfg)
        except Exception as e:
            raise RunnerInitializationError(f"Failed to create runner with config: {e}") from e


# --- Execution Strategies ---

class IExecutionStrategy(ABC):
    """Abstract base class for execution workflows."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.runner = RunnerFactory.create(cfg)

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """Execute the workflow."""
        pass


class TrainingStrategy(IExecutionStrategy):
    """Handles the model training workflow."""
    
    def execute(self, **kwargs) -> None:
        logger.info("Initializing Training Workflow...")
        self.runner.train(cfg=self.cfg)
        logger.info("Training Workflow Completed.")


class EvaluationStrategy(IExecutionStrategy):
    """Handles the model evaluation/testing workflow."""
    
    def execute(self, ckpt_path: Optional[str] = None, strict: bool = True) -> None:
        logger.info("Initializing Evaluation Workflow...")
        
        # Resolve Checkpoint
        resolver = CheckpointResolver(self.runner)
        resolved_path = resolver.resolve(ckpt_path)
        
        # Load Model
        logger.info(f"Loading weights from {resolved_path}")
        self.runner.load_model(ckpt_path=str(resolved_path))
        
        # Run Pipeline
        logger.info("Starting Test Pipeline...")
        self.runner.test_pipeline(cfg=self.cfg)
        logger.info("Evaluation Workflow Completed.")


# --- Public Entry Points (Facades) ---

def launch_evaluation(cfg: DictConfig, ckpt_path: str, strict: bool = True) -> None:
    """
    Public API to launch the evaluation process.
    
    Args:
        cfg (DictConfig): Configuration object.
        ckpt_path (str): Path to the checkpoint.
        strict (bool): Strict mode flag (reserved for future use).
    """
    executor = EvaluationStrategy(cfg)
    executor.execute(ckpt_path=ckpt_path, strict=strict)


def launch_training(cfg: DictConfig) -> None:
    """
    Public API to launch the training process.
    
    Args:
        cfg (DictConfig): Configuration object.
    """
    executor = TrainingStrategy(cfg)
    executor.execute()