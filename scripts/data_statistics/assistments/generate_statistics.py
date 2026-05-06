import os
from itertools import chain
from typing import Dict

import pandas as pd
from loguru import logger
from tqdm import tqdm


def before_process(data_path: str, dataset_name: str):
    """Perform basic preprocessing and descriptive statistics on `ASSISTments` datasets.

    This function provides initial statistics for the following datasets:
    - ASSISTmemts2009-2010
    - ASSISTments2012-2013
    - ASSISTments2015
    - ASSISTments2017

    Args:
        data_path (str): The root path of the data.
        dataset_name (str): Name of the dataset.
    """

    raw_data_path = os.path.join(data_path, dataset_name, "raw_data")
    dataset_info = get_dataset_info(dataset_name=dataset_name)

    interactions = 0
    users = set()
    questions = set()
    knowledge_components = set()

    # Option 1:
    total_lines = 0
    for chunk in pd.read_csv(os.path.join(raw_data_path, dataset_info.get("log_filename")),
                             dtype=str, chunksize=100000, encoding="ISO-8859-15"):
        total_lines += len(chunk)
    pbar = tqdm(total=total_lines, desc=f"Processing {dataset_name} chunks")

    # Option 2:
    # pbar = tqdm(desc=f"Processing {dataset_name} chunks")

    with pbar:
        for chunk in pd.read_csv(os.path.join(raw_data_path, dataset_info.get("log_filename")),
                                 dtype=str, usecols=list(dataset_info.get("log_input_columns").keys()),
                                 chunksize=100000, encoding="ISO-8859-15"):
            chunk = chunk.rename(columns=dataset_info.get("log_input_columns"))

            interactions += len(chunk)
            users.update(chunk["user_id"].dropna())
            questions.update(chunk["question_id"].dropna())

            if "knowledge_component_id" in chunk.columns:
                kc_series = chunk["knowledge_component_id"].dropna().str.split("_")
                knowledge_components.update(chain.from_iterable(kc_series))

            pbar.update(len(chunk))

    logger.info(f"------------------------ Before Processing ({dataset_name}) -----------------------")
    logger.info(f"#Interactions: {interactions:,}")
    logger.info(f"#Students: {len(users):,}")
    logger.info(f"#Questions: {len(questions):,}")
    logger.info(f"#KCs: {len(knowledge_components):,}")


def after_process(data_path: str, dataset_name: str, min_user_inter_num: int):
    """Perform post-processing and descriptive statistics on `ASSISTments` datasets.

    Processing steps:
        1. Load selected columns from the dataset.
        2. Remove rows with any missing values.
        3. Remove duplicate rows.
        4. Keep only binary labels in `correct` (0 or 1).
        5. Filter out users with fewer than `min_user_inter_num` interactions.

    Statistics computed:
        - Total number of interactions
        - Number of unique students
        - Number of unique questions
        - Number of unique knowledge components (KCs)
        - Positive label rate
        - Average interactions per student
        - Average interactions per question
        - Average KCs per question

    Args:
        data_path (str): The root path of the data.
        dataset_name (str): Name of the dataset.
        min_user_inter_num (int): Minimum number of interactions per student to keep.
    """

    raw_data_path = os.path.join(data_path, dataset_name, "raw_data")
    dataset_info = get_dataset_info(dataset_name=dataset_name)

    interactions = 0
    users = set()
    questions = set()
    total_knowledge_components = 0
    knowledge_components = set()
    knowledge_components_recode = set()
    positive_label_count = 0

    log_df = pd.read_csv(os.path.join(raw_data_path, dataset_info.get("log_filename")),
                         dtype=str, usecols=list(dataset_info.get("log_input_columns").keys()),
                         encoding="ISO-8859-15")

    # Rename columns
    log_df = log_df.rename(columns=dataset_info.get("log_input_columns"))
    # Remove NaN
    log_df = log_df.dropna(axis=0, how="any")
    # Remove duplicates
    log_df = log_df.drop_duplicates()
    # Keep only binary labels
    log_df = log_df[log_df["response"].isin(["0", "1"])]
    # Filter too short sequences
    log_df = log_df.groupby("user_id").filter(lambda x: len(x) >= min_user_inter_num)

    interactions += len(log_df)
    users.update(log_df["user_id"].dropna())
    questions.update(log_df["question_id"].dropna())

    if "knowledge_component_id" in log_df.columns:
        knowledge_components_recode.update(log_df["knowledge_component_id"].dropna())
        kc_series = log_df["knowledge_component_id"].dropna().str.split("_")
        total_knowledge_components += kc_series.str.len().sum()
        knowledge_components.update(chain.from_iterable(kc_series))

    positive_label_count += (log_df["response"] == "1").sum()

    logger.info(f"------------------------ After Processing ({dataset_name}) -----------------------")
    logger.info(f"#Interactions: {interactions:,}")
    logger.info(f"#Students: {len(users):,}")
    logger.info(f"#Questions: {len(questions):,}")
    logger.info(f"#KCs: {len(knowledge_components):,}")
    logger.info(f"#KCs*: {len(knowledge_components_recode):,}")
    logger.info(f"Positive Label Rate: {(positive_label_count / interactions * 100) if interactions > 0 else 0:.2f}%")
    logger.info(f"Average Interactions per Student: {(interactions / len(users)) if len(users) > 0 else 0:,.2f}")
    logger.info(
        f"Average Interactions per Question: {(interactions / len(questions)) if len(questions) > 0 else 0:,.2f}")
    logger.info(
        f"Average KCs per Question: {(total_knowledge_components / interactions) if interactions > 0 else 0:.2f}")


def get_dataset_info(dataset_name: str) -> Dict:
    """Return configuration for `ASSISTments` datasets.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        dataset_info (dict): Configuration dictionary containing data settings.
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
            }
        },
        "assist2012": {
            "log_filename": "2012-2013-data-with-predictions-4-final.csv",
            "log_input_columns": {
                "problem_log_id": "log_id",
                "user_id": "user_id",
                "problem_id": "question_id",
                "skill_id": "knowledge_component_id",
                "correct": "response"
            }
        },
        "assist2015": {
            "log_filename": "2015_100_skill_builders_main_problems.csv",
            "log_input_columns": {
                "log_id": "log_id",
                "user_id": "user_id",
                "sequence_id": "question_id",
                "correct": "response"
            }
        },
        "assist2017": {
            "log_filename": "anonymized_full_release_competition_dataset.csv",
            "log_input_columns": {
                "action_num": "log_id",
                "studentId": "user_id",
                "problemId": "question_id",
                "skill": "knowledge_component_id",
                "correct": "response"
            }
        }
    }

    if dataset_name not in dataset_info:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(dataset_info.keys())}"
        )

    return dataset_info[dataset_name]


def main(data_path: str, dataset_name: str, min_user_inter_num: int):
    """Data statistics pipeline.

    Args:
        data_path (str): The root path of the data.
        dataset_name (str): Name of the dataset.
        min_user_inter_num (int): Minimum number of interactions per student to keep.
    """

    before_process(data_path=data_path, dataset_name=dataset_name)
    after_process(data_path=data_path, dataset_name=dataset_name, min_user_inter_num=min_user_inter_num)
