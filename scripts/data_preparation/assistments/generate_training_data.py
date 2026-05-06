import json
import os
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pandas import DataFrame

from scripts.data_preparation.utils import pad_sequence, replace_text, split_data


def get_dataset_info(dataset_name: str) -> Dict:
    """Return information for `ASSISTments` datasets.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Dict: Dataset information.
    """

    dataset_info = {
        "assist2009": {
            "log_filename": "skill_builder_data_corrected_collapsed.csv",
            "log_input_columns": {
                "order_id": "log_id",
                "user_id": "user_id",
                "problem_id": "question_id",
                "skill_id": "knowledge_component_id",
                "correct": "response"
            },
            "log_output_columns": ["question_id", "knowledge_component_id", "response"],
            "additional_features": [],
            "log_recode_columns": ["question_id", "knowledge_component_id"]
        },
        "assist2012": {
            "log_filename": "2012-2013-data-with-predictions-4-final.csv",
            "log_input_columns": {
                "problem_log_id": "log_id",
                "user_id": "user_id",
                "problem_id": "question_id",
                "skill_id": "knowledge_component_id",
                "correct": "response"
            },
            "log_output_columns": ["question_id", "knowledge_component_id", "response"],
            "additional_features": [],
            "log_recode_columns": ["question_id", "knowledge_component_id"]
        },
        "assist2015": {
            "log_filename": "2015_100_skill_builders_main_problems.csv",
            "log_input_columns": {
                "log_id": "log_id",
                "user_id": "user_id",
                "sequence_id": "question_id",
                "correct": "response"
            },
            # To prevent errors, use `question_id` as `knowledge_component_id`
            "log_output_columns": ["question_id", "knowledge_component_id", "response"],
            "additional_features": [],
            "log_recode_columns": ["question_id", "knowledge_component_id"]
        },
        "assist2017": {
            "log_filename": "anonymized_full_release_competition_dataset.csv",
            "log_input_columns": {
                "action_num": "log_id",
                "studentId": "user_id",
                "problemId": "question_id",
                "skill": "knowledge_component_id",
                "correct": "response"
            },
            "log_output_columns": ["question_id", "knowledge_component_id", "response"],
            "additional_features": [],
            "log_recode_columns": ["question_id", "knowledge_component_id"]
        }
    }

    if dataset_name not in dataset_info:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(dataset_info.keys())}"
        )

    return dataset_info[dataset_name]


def load_and_preprocess_data(data_cfg: DictConfig) -> DataFrame:
    """Preprocess the raw data into a standard format suitable for sequence-based modeling.

    Args:
        data_cfg (DictConfig): Configuration dictionary containing data settings.
        
    Returns:
        DataFrame: Preprocessed data.
    """

    dataset_info = get_dataset_info(dataset_name=data_cfg.DATASET_NAME)
    raw_data_path = os.path.join(data_cfg.ROOT_PATH, data_cfg.DATASET_NAME, "raw_data")

    log_df = pd.read_csv(os.path.join(raw_data_path, dataset_info.get("log_filename")),
                         dtype=str, usecols=list(dataset_info.get("log_input_columns").keys()),
                         encoding="ISO-8859-15")

    log_df = (
        log_df
        .rename(columns=dataset_info.get("log_input_columns"))  # Rename columns
        .dropna(axis=0, how="any")  # Remove NaN
        .drop_duplicates()  # Remove duplicates
        .assign(
            log_id=lambda x: pd.to_numeric(x["log_id"], errors="coerce"),
            knowledge_component_id=lambda x: x["knowledge_component_id"]
            if "knowledge_component_id" in x.columns
            else x["question_id"]
        )
        .sort_values(by="log_id", ascending=True)  # Sort by time
        .reset_index(drop=True)
        .query("response in ['0', '1']")  # Keep only binary labels
    )

    # Replace special characters
    log_df = replace_text(data=log_df)

    # Convert to sequence
    grouped_data = (
        log_df
        .groupby("user_id")[dataset_info.get("log_output_columns")]
        .agg(lambda x: ",".join(x.astype(str)))
        .reset_index()
        .drop(columns="user_id")
    )
    grouped_data = (
        grouped_data
        .sample(n=min(data_cfg.SAMPLE, len(grouped_data)), random_state=data_cfg.RANDOM_STATE)
        .reset_index(drop=True)
    )

    unique_counts = {}
    for col in grouped_data.columns:
        all_values = set()
        for seq in grouped_data[col]:
            all_values.update(seq.split(","))
        unique_counts[col] = len(all_values)

    logger.info(
        f"Loaded and preprocessed data: {len(grouped_data):,} users, "
        f"{len(grouped_data.columns)} features ({'; '.join(f'{col}: {count:,}' for col, count in unique_counts.items())})"
    )

    return grouped_data


def add_features(data: DataFrame, data_cfg: DictConfig) -> DataFrame:
    """Enhance the data by adding additional features, such as `Difficulty` and `Time`.

    Args:
        data (DataFrame): Preprocessed data with sequences for each user,
                          as returned by `load_and_preprocess_data`.
        data_cfg (DictConfig): Configuration dictionary containing data settings.

    Returns:
        DataFrame: Enhanced data.
    """

    dataset_info = get_dataset_info(dataset_name=data_cfg.DATASET_NAME)
    enhanced_data = data.copy()

    added_features = set(enhanced_data.columns) - set(data.columns)
    logger.info(
        f"Enhanced data with additional features: ({', '.join(added_features)})"
    )

    return enhanced_data


def recode_data(data: DataFrame, data_cfg: DictConfig) -> Tuple[Dict[str, dict], DataFrame]:
    """Re-encode categorical columns (e.g., question IDs) into consecutive integers.
    Each unique value is assigned a unique integer index.

    Args:
        data (DataFrame): Enhanced data.
        data_cfg (DictConfig): Configuration dictionary containing data settings.

    Returns:
        Tuple[Dict[str, dict], DataFrame]: Mapping data and Recoded data.
    """

    dataset_info = get_dataset_info(dataset_name=data_cfg.DATASET_NAME)

    map_json = {}
    recoded_data = data.copy()

    for key in dataset_info.get("log_recode_columns"):
        col_data = data[key].fillna("")

        mask = col_data != ""
        split_data = col_data[mask].str.split(",")

        all_values = set()
        for item_list in split_data:
            all_values.update(item_list)

        sorted_values = sorted(all_values)
        mapping = {qid: index for index, qid in enumerate(sorted_values)}
        str_mapping = {qid: str(index) for index, qid in enumerate(sorted_values)}
        map_json[key] = mapping

        # Vectorized recoding
        def recode_row(row_data):
            if row_data == "":
                return ""
            items = row_data.split(",")
            return ",".join(str_mapping[item] for item in items)

        recoded_col = col_data.copy()
        recoded_col[mask] = col_data[mask].apply(recode_row)
        recoded_data[key] = recoded_col

    logger.info(
        f"Recoded {len(dataset_info.get('log_recode_columns'))} columns with mappings: "
        f"({', '.join(map_json.keys())})"
    )

    return map_json, recoded_data


def main(data_cfg: DictConfig, model_name: str):
    """Data preprocessing pipeline.

    Args:
        data_cfg (DictConfig): Configuration dictionary containing data settings.
        model_name (str): Name of the model.
    """

    logger.info("=" * 30 + " PREPARING DATA " + "=" * 30)

    data = load_and_preprocess_data(data_cfg=data_cfg)

    enhanced_data = add_features(data=data, data_cfg=data_cfg)

    map_json, recoded_data = recode_data(data=enhanced_data, data_cfg=data_cfg)

    sequence_data = pad_sequence(data=recoded_data, data_cfg=data_cfg)

    train_data, valid_data, test_data = split_data(data=sequence_data, data_cfg=data_cfg)

    # Save processed data
    save_path = os.path.join(data_cfg.ROOT_PATH, data_cfg.DATASET_NAME, "model_data", model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, data_cfg.FILES.MAP_JSON.NAME), "w") as json_file:
        json.dump(map_json, json_file, indent=4)

    train_data.to_csv(os.path.join(save_path, data_cfg.FILES.TRAIN_DATA.NAME), index=False)
    valid_data.to_csv(os.path.join(save_path, data_cfg.FILES.VALID_DATA.NAME), index=False)
    test_data.to_csv(os.path.join(save_path, data_cfg.FILES.TEST_DATA.NAME), index=False)
