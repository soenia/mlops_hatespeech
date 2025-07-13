from pathlib import Path

import typer
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_PATH = ROOT_DIR / "data" / "processed"


def load_and_prepare_dataset(seed: int = 42, save_path: str = None):
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    else:
        save_path = Path(save_path)

    # 1. Load original dataset
    ds = load_dataset("thefrankhsu/hate_speech_twitter")
    
    # 2. Concatenate train and test into one dataset
    combined = concatenate_datasets([ds["train"], ds["test"]])
    
    # 3. Shuffle for good measure
    combined = combined.shuffle(seed=seed)

    # 4. Split into train (70%), val (15%), test (15%)
    n = len(combined)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_ds = combined.select(range(0, train_size))
    val_ds = combined.select(range(train_size, train_size + val_size))
    test_ds = combined.select(range(train_size + val_size, n))

    # 5. Combine into DatasetDict and save
    full_dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    full_dataset.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    typer.run(load_and_prepare_dataset)
