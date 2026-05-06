import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, List, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Configuration & Logging ---
logger = logging.getLogger(__name__)

# --- Type Definitions ---
TensorType = torch.Tensor
ArrayType = np.ndarray
MetricValue = torch.Tensor

# --- Data Processor Component ---

class TensorSanitizer:
    """
    Responsible for sanitizing and converting raw model outputs and targets
    into a format suitable for Scikit-Learn metrics.
    """

    @staticmethod
    def ensure_mask(prediction: TensorType, mask: Optional[TensorType]) -> TensorType:
        """Ensures a valid boolean mask exists."""
        if mask is None:
            return torch.ones_like(prediction, dtype=torch.bool)
        return mask.bool()

    @staticmethod
    def apply_mask_and_convert(
        prediction: TensorType, 
        target: TensorType, 
        mask: TensorType
    ) -> Tuple[ArrayType, ArrayType]:
        """
        Applies the mask to flatten the tensors and converts them to CPU NumPy arrays.
        
        Args:
            prediction: Prediction tensor.
            target: Target tensor.
            mask: Boolean mask tensor.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Flattened masked arrays (pred, target).
        """
        # Select masked elements
        masked_preds = torch.masked_select(prediction, mask)
        masked_targets = torch.masked_select(target, mask)

        # Detach and convert to numpy
        np_preds = masked_preds.detach().cpu().numpy()
        np_targets = masked_targets.detach().cpu().numpy()

        return np_preds, np_targets


# --- Abstract Base Metric Strategy ---

class BaseMaskedMetric(ABC):
    """
    Abstract base class defining the workflow for calculating metrics
    on masked sequence data.
    """
    
    def __init__(self, default_value: float = 0.5):
        self.default_value = default_value
        self.sanitizer = TensorSanitizer()

    def __call__(
        self, 
        prediction: TensorType, 
        target: TensorType, 
        mask: Optional[TensorType] = None
    ) -> MetricValue:
        """
        Orchestrates the metric calculation pipeline.
        """
        # 1. Prepare Mask
        valid_mask = self.sanitizer.ensure_mask(prediction, mask)
        
        # 2. Sanitize Data
        y_pred, y_true = self.sanitizer.apply_mask_and_convert(prediction, target, valid_mask)
        
        # 3. Validation (Optional hook for subclasses)
        if not self._validate_inputs(y_true, y_pred):
            logger.debug(f"Input validation failed for {self.__class__.__name__}, returning default.")
            return torch.tensor(self.default_value)

        # 4. Compute Specific Metric
        try:
            score = self._compute_core_metric(y_true, y_pred)
            return torch.tensor(score, dtype=torch.float64 if isinstance(score, float) else torch.float32) # match original implicit types
        except ValueError as e:
            logger.warning(f"Metric calculation error in {self.__class__.__name__}: {e}")
            return torch.tensor(self.default_value)

    def _validate_inputs(self, y_true: ArrayType, y_pred: ArrayType) -> bool:
        """Default validation checks for empty inputs."""
        if len(y_true) == 0:
            return False
        return True

    @abstractmethod
    def _compute_core_metric(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Implementation of the actual scikit-learn metric logic."""
        raise NotImplementedError


# --- Concrete Metric Implementations ---

class AccuracyMetric(BaseMaskedMetric):
    """Calculates binary accuracy with a 0.5 threshold."""

    def __init__(self, threshold: float = 0.5):
        super().__init__(default_value=0.0) # Accuracy default usually 0 if empty, though code implicitly didn't handle empty acc distinct from error
        self.threshold = threshold

    def _compute_core_metric(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        # Binarize predictions
        binary_preds = (y_pred >= self.threshold).astype(int)
        return float(accuracy_score(y_true=y_true, y_pred=binary_preds))


class AUCMetric(BaseMaskedMetric):
    """Calculates Area Under the ROC Curve."""

    def _validate_inputs(self, y_true: ArrayType, y_pred: ArrayType) -> bool:
        # Parent check (empty)
        if not super()._validate_inputs(y_true, y_pred):
            return False
        
        # AUC specific check: Must have at least 2 classes to compute ROC
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            return False
        
        return True

    def _compute_core_metric(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        return float(roc_auc_score(y_true=y_true, y_score=y_pred))


# --- Functional Facades (Exposed API) ---

_acc_evaluator = AccuracyMetric()
_auc_evaluator = AUCMetric()

def masked_acc(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes accuracy on masked tensors.
    
    Args:
        prediction (torch.Tensor): Logits or probabilities.
        target (torch.Tensor): Ground truth labels.
        mask (torch.Tensor, optional): Boolean mask indicating valid indices.
        
    Returns:
        torch.Tensor: Scalar tensor containing the accuracy.
    """
    return _acc_evaluator(prediction, target, mask)


def masked_auc(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes AUC-ROC on masked tensors. Handles edge cases where only one class is present.
    
    Args:
        prediction (torch.Tensor): Probabilities.
        target (torch.Tensor): Ground truth labels.
        mask (torch.Tensor, optional): Boolean mask indicating valid indices.
        
    Returns:
        torch.Tensor: Scalar tensor containing the AUC score (or 0.5 if undefined).
    """
    return _auc_evaluator(prediction, target, mask)


__all__ = ["masked_auc", "masked_acc"]