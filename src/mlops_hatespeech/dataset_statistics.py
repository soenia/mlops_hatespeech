from datasets import load_from_disk
import matplotlib.pyplot as plt
from pathlib import Path
import typer
import torch


def compute_statistics(datadir: str = "data/processed") -> None:
    """Compute dataset statistics."""
    dataset = load_from_disk(datadir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    validation_dataset = dataset["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    train_label_distribution = torch.bincount(torch.tensor(train_dataset["label"], dtype=torch.int64))
    test_label_distribution = torch.bincount(torch.tensor(test_dataset["label"], dtype=torch.int64))
    validation_label_distribution = torch.bincount(torch.tensor(validation_dataset["label"], dtype=torch.int64))

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.bar(torch.arange(2), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(figures_dir / f"train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(2), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(figures_dir / f"test_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(2), validation_label_distribution)
    plt.title("Validation label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(figures_dir / f"validation_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(compute_statistics)
