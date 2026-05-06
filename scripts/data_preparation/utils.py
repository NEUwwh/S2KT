from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def replace_text(data: DataFrame) -> pd.DataFrame:
    """Replace special characters in string columns of a DataFrame.

    Args:
        data (DataFrame): Input pandas DataFrame.

    Returns:
        DataFrame
    """

    for col in data.columns:
        if data[col].dtype == object or pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].astype(str).map(lambda x: x.replace(",", "@@@@"))

    return data


def pad_sequence(data: DataFrame, data_cfg: DictConfig) -> DataFrame:
    """Segment and pad user sequences into fixed-length input chunks.

    sequences shorter than `MIN_SEQ_LENGTH` are discarded,
    sequences longer than `MAX_SEQ_LENGTH` are split into multiple chunks,
    shorter chunks are padded with a special `PAD_VAL` token.

    Args:
        data (DataFrame): Log file.
        data_cfg (DictConfig): Configuration dictionary containing data settings.

    Returns:
        DataFrame: Sequence file.
    """

    min_len = data_cfg.MIN_SEQ_LENGTH
    max_len = data_cfg.MAX_SEQ_LENGTH

    def pad_and_split(seq: List[str]) -> List[List[str]]:
        if len(seq) < min_len:
            return []
        chunks = [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
        if len(chunks[-1]) < min_len:
            chunks.pop()
        for chunk in chunks:
            if len(chunk) < max_len:
                chunk += [data_cfg.PAD_VAL] * (max_len - len(chunk))

        return chunks

    sequence_records: Dict[str, List] = {col: [] for col in data.columns}
    sequence_records["select_mask"] = []

    for _, row in data.iterrows():
        raw_sequences = {key: row[key].split(",") for key in data.columns}
        processed = {key: pad_and_split(raw_sequences[key]) for key in data.columns}

        if any(len(v) == 0 for v in processed.values()):
            continue

        num_chunks = len(next(iter(processed.values())))
        for i in range(num_chunks):
            for key in data.columns:
                sequence_records[key].append(",".join(processed[key][i]))

            pad_count = next(iter(processed.values()))[i].count(data_cfg.PAD_VAL)
            mask = ["1"] * (max_len - pad_count) + ["-1"] * pad_count
            sequence_records["select_mask"].append(",".join(mask))

    logger.info(
        f"Padded sequences to fixed length range (min={min_len}, max={max_len}); "
        f"total sequences: {len(next(iter(sequence_records.values()))):,}"
    )

    return DataFrame(sequence_records)


def split_data(data: DataFrame, data_cfg: DictConfig) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split the log data.

    splits the data into training-validation and testing sets using `TEST_RATIO`,
    splits training-validation set into `K_FOLD` folds,
        - fold ∈ {0, ..., K_FOLD-1} - {VALID_FOLD} for train set,
        - fold = `VALID_FOLD` for valid set,
        - fold = `TEST_FOLD` for test set.

    A new field `fold` is added to indicate sets:
    fold ∈ {0, ..., K_FOLD-1, TEST_FOLD}

    Args:
        data (DataFrame): Log file.
        data_cfg (DictConfig): Configuration dictionary containing data settings.

    Returns:
        DataFrame: Training file.
        DataFrame: Validation file
        DataFrame: Testing file.
    """

    train_valid_data, test_data = train_test_split(data, test_size=data_cfg.TEST_RATIO,
                                                   random_state=data_cfg.RANDOM_STATE)
    train_valid_splits = np.array_split(train_valid_data, data_cfg.K_FOLD)
    for i, split in enumerate(train_valid_splits):
        split["fold"] = i
    test_data["fold"] = data_cfg.TEST_FOLD

    train_valid_data = pd.concat(train_valid_splits)
    train_data = train_valid_data[train_valid_data["fold"] != data_cfg.VALID_FOLD]
    valid_data = train_valid_data[train_valid_data["fold"] == data_cfg.VALID_FOLD]

    train_data = train_data.drop(columns="fold")
    valid_data = valid_data.drop(columns="fold")
    test_data = test_data.drop(columns="fold")

    logger.info(
        f"Split data into train/valid/test sets: "
        f"{len(train_data)} ({len(train_data) / len(data):.1%}), "
        f"{len(valid_data)} ({len(valid_data) / len(data):.1%}), "
        f"{len(test_data)} ({len(test_data) / len(data):.1%})"
    )

    return train_data, valid_data, test_data
