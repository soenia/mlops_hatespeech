import os

from datasets import load_from_disk
from mlops_hatespeech.data import load_and_prepare_dataset


def test_data(tmp_path):
    """Test loading and preparing the dataset."""

    load_and_prepare_dataset(split_val=0.2, seed=42, save_path=tmp_path)
    assert os.path.exists(tmp_path)
    dataset = load_from_disk(str(tmp_path))

    # Check expected splits
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset

    # Check dataset sizes
    assert len(dataset["train"]) > 0
    assert len(dataset["validation"]) > 0
    assert len(dataset["test"]) > 0

    # Check a sample has expected fields
    sample = dataset["train"][0]
    assert "tweet" in sample
    assert "label" in sample

    # Check dataset types
    assert dataset["train"].features["tweet"].dtype == "string"
    assert dataset["train"].features["label"].dtype == "int64"
