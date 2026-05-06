from omegaconf import DictConfig

from .assistments import generate_assistments_data


DATASET_FUNCTION_MAP = {
    "assist2009": generate_assistments_data,
}


def data_preparation_pipeline(data_cfg: DictConfig, dataset_name: str, model_name: str):
    """Data preprocessing pipeline.

    Args:
        data_cfg (DictConfig): Configuration dictionary containing data settings.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
    """

    if dataset_name not in DATASET_FUNCTION_MAP:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    DATASET_FUNCTION_MAP[dataset_name](data_cfg=data_cfg, model_name=model_name)
