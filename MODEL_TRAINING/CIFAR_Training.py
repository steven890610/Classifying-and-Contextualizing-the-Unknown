from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import torch
import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
SEED = 42
BEST_CKPT = "vit_cifar10_first4_best.pt"
SAVE_DIR = "./vit_cifar10_first4_model"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Dataset filters
# -----------------------------
def filter_first_four_labels(example):
    return example["label"] < 4

# -----------------------------
# Load CIFAR datasets
# -----------------------------
# CIFAR-10 for train/val/test (ID classes)
train_ds, test_ds_cifar10 = load_dataset("cifar10", split=["train[:]", "test[:]"])

# CIFAR-100 for OOD testing later if needed
_, test_ds_cifar100 = load_dataset("cifar100", split=["train[:]", "test[:]"])

# Split CIFAR-10 train into train/val
splits = train_ds.train_test_split(test_size=0.1, seed=SEED)
train_ds = splits["train"]
val_ds = splits["test"]

# Keep only labels 0,1,2,3 from CIFAR-10
train_ds = train_ds.filter(filter_first_four_labels)
val_ds = val_ds.filter(filter_first_four_labels)
test_ds_cifar10_4 = test_ds_cifar10.filter(filter_first_four_labels)

# -----------------------------
# Label mappings for the 4 kept classes
# -----------------------------
all_names = train_ds.features["label"].names
class_names_4 = all_names[:4]

id2label_4 = {i: class_names_4[i] for i in range(4)}
label2id_4 = {class_names_4[i]: i for i in range(4)}

print("id2label_4:", id2label_4)
print("label2id_4:", label2id_4)

# -----------------------------
# Processor
# -----------------------------
# Since the node has internet, this can download and cache automatically
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"] if isinstance(processor.size, dict) else processor.size

normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

_val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["img"]]
    return examples

def val_transforms(examples):
    examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["img"]]
    return examples

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds_cifar10_4.set_transform(val_transforms)
test_ds_cifar100.set_transform(val_transforms)

# -----------------------------
# Dataloaders
# -----------------------------
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(
    train_ds,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=EVAL_BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds_cifar10_4,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=EVAL_BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
)

ood_loader = DataLoader(
    test_ds_cifar100,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=EVAL_BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
)

# Quick sanity check
batch = next(iter(train_loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

# -----------------------------
# Model
# -----------------------------
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=id2label_4,
    label2id=label2id_4,
    ignore_mismatched_sizes=True,
)
model.to(DEVICE)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total

@torch.no_grad()
def collect_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.detach().cpu()

        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels

# -----------------------------
# Training loop
# -----------------------------
best_val_acc = -1.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, batch in enumerate(train_loader, start=1):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)

        if step % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Train Loss: {running_loss / running_total:.4f} | "
                f"Train Acc: {running_correct / running_total:.4f}"
            )

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc = evaluate(model, val_loader, DEVICE)

    print(
        f"\nEpoch {epoch + 1} complete | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"Saved best checkpoint to {BEST_CKPT} with val_acc={val_acc:.4f}")
# -----------------------------
# Save full Hugging Face model + processor
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"Saved fine-tuned model and processor to: {SAVE_DIR}")

'''
apptainer exec --nv \
  --env HF_DATASETS_OFFLINE=0,HF_HUB_OFFLINE=0,TRANSFORMERS_OFFLINE=0 \
  --env HF_HOME=/projects/academic/kylehunt/hf_cache \
  --env HF_DATASETS_CACHE=/projects/academic/kylehunt/hf_cache/datasets \
  --env HUGGINGFACE_HUB_CACHE=/projects/academic/kylehunt/hf_cache/hub \
  --bind /projects/academic/kylehunt/clean_osr:/projects/academic/kylehunt/clean_osr \
  --bind /projects/academic/kylehunt/hf_cache:/projects/academic/kylehunt/hf_cache \
  --pwd /projects/academic/kylehunt/clean_osr \
  "$SIF" \
  python cifar_plus_finetune.py
'''