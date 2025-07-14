import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from mlops_hatespeech.model import MODEL_STR
from mlops_hatespeech.train import train_model


def test_model_instantiation():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_STR,
        num_labels=2,
        id2label={0: "non-hate", 1: "hate"},
        label2id={"non-hate": 0, "hate": 1},
    )
    assert isinstance(model, torch.nn.Module)
    assert model.config.num_labels == 2


def test_train_model(tmp_path):
    # Create a dummy dataset
    data = {"tweet": ["Hate speech", "Nice words"], "label": [1, 0]}
    ds = Dataset.from_dict(data)
    ds_dict = DatasetDict({"train": ds, "validation": ds})

    dataset_path = tmp_path / "dummy_data"
    ds_dict.save_to_disk(str(dataset_path))

    cfg = OmegaConf.create(
        {
            "data_path": str(dataset_path),
            "hyperparameters": {
                "lr": 2e-5,
                "epochs": 1,
                "seed": 42,
                "data_path": str(dataset_path),
                "output_dir": str(tmp_path / "outputs"),
                "wd": 0.01,
                "per_device_train_batch_size": 2,
                "per_gpu_eval_batch_size": 2,
                "logging_strategy": "steps",
                "logging_steps": 10,
                "save_strategy": "no",
                "eval_strategy": "no",
                "eval_steps": 100,
                "save_total_limit": 1,
                "load_best_model_at_end": False,
                "report_to": None,
                "use_cpu": True,
                "dataloader_num_workers": 0,
                "gradient_accumulation_steps": 2,
            },
        }
    )

    train_model(cfg)
