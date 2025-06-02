from pathlib import Path

from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import typer
from datasets import load_dataset, DatasetDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_PATH = ROOT_DIR / "data" / "processed"

def load_and_prepare_dataset(
    split_val: float = 0.25,
    seed: int = 42,
    save_path: str = None
):
    if save_path is None:
        save_path = DEFAULT_SAVE_PATH
    else:
        save_path = Path(save_path)

    # 1. Load dataset
    ds = load_dataset("thefrankhsu/hate_speech_twitter")

    # 2. Split train into train + val
    train_valid = ds["train"].train_test_split(test_size=split_val, seed=seed)

    # 3. Recombine
    full_dataset = DatasetDict({
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": ds["test"]
    })

    # 4. Save
    full_dataset.save_to_disk(str(save_path))
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    typer.run(load_and_prepare_dataset)
