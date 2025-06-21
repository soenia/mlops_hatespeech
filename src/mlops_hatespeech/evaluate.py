import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk
from mlops_hatespeech.model import MODEL_STR

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_STR)
model.eval()

# Load test dataset
test_dataset = load_from_disk("data/processed")["test"]

# find and use correct label
label_key = None
sample_keys = list(test_dataset[0].keys())
for key in sample_keys:
    if key.lower() in ["label", "labels", "target", "y"]:
        label_key = key
        break

if label_key is None:
    print(f"Could not find a label key in the dataset. Available keys: {sample_keys}")
    exit(1)

print(f"Using '{label_key}' as the label key.")

# Store predictions and labels
all_preds = []
all_labels = []

for example in test_dataset:
    inputs = tokenizer(
        example["tweet"],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items()})
        pred = torch.argmax(outputs.logits, dim=1).item()
    all_preds.append(pred)
    all_labels.append(example[label_key])

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")