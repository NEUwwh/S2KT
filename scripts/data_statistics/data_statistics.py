from .assistments import generate_assistments_statistics



DATASET_FUNCTION_MAP = {
    "assist2009": generate_assistments_statistics,
}


def data_statistics_pipeline(data_path: str, dataset_name: str, min_user_inter_num: int):
    """Data statistics pipeline.

    Args:
        data_path (str): The root path of the data.
        dataset_name (str): Name of the dataset.
        min_user_inter_num (int): Minimum number of interactions per student to keep.
    """

    if dataset_name not in DATASET_FUNCTION_MAP:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    DATASET_FUNCTION_MAP[dataset_name](
        data_path=data_path,
        dataset_name=dataset_name,
        min_user_inter_num=min_user_inter_num
    )
