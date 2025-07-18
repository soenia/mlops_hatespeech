"""
Script to download and prepare the hate speech Twitter dataset.

- Downloads the dataset from Hugging Face (`thefrankhsu/hate_speech_twitter`)
- Merges train and test splits into a single dataset
- Shuffles and splits the data into train (70%), validation (15%), and test (15%)
- Saves the resulting `DatasetDict` to disk for later use
"""

from pathlib import Path
from typing import Optional

import typer
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from mlops_hatespeech.logger import logger


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_PATH = ROOT_DIR / "data" / "processed"


def load_and_prepare_dataset(seed: int = 42, save_path: Optional[str] = None) -> None:
    """
    Loads, merges, splits, and saves the hate speech dataset.

    Args:
        seed (int): Random seed for shuffling the dataset.
        save_path (Optional[str]): Path to save the processed dataset.
        If None, uses default path under /data/processed.

    Returns:
        None
    """
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    else:
        save_path = Path(save_path)

    # Load original dataset
    ds = load_dataset("thefrankhsu/hate_speech_twitter")

    # Concatenate train and test into one dataset
    combined = concatenate_datasets([ds["train"], ds["test"]])

    # Shuffle for good measure
    combined = combined.shuffle(seed=seed)

    # Split into train (70%), val (15%), test (15%)
    n = len(combined)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_ds = combined.select(range(0, train_size))
    val_ds = combined.select(range(train_size, train_size + val_size))
    test_ds = combined.select(range(train_size + val_size, n))

    # Combine into DatasetDict and save
    full_dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    # Concatenate train and test into one dataset
    combined = concatenate_datasets([ds["train"], ds["test"]])

    # Shuffle for good measure
    combined = combined.shuffle(seed=seed)

    # Split into train (70%), val (15%), test (15%)
    n = len(combined)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_ds = combined.select(range(0, train_size))
    val_ds = combined.select(range(train_size, train_size + val_size))
    test_ds = combined.select(range(train_size + val_size, n))

    # Combine into DatasetDict and save
    full_dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    full_dataset.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    typer.run(load_and_prepare_dataset)
