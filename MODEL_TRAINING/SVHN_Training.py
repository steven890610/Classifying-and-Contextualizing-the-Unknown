import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# -----------------------------
# Optional stability settings
# -----------------------------
# Uncomment if your container has CUDA issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ID / OOD split
ID_CLASSES = [0, 1, 2, 3, 4, 5]
NUM_CLASSES = len(ID_CLASSES)

# WideResNet config
DEPTH = 16            # try 28 for WRN-28-x
WIDEN_FACTOR = 4      # try 10 for WRN-28-10
DROPOUT_RATE = 0.0

# Training config
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 5e-4
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

SAVE_DIR = "./svhn_wrn_0_5_model"
BEST_CKPT = "./svhn_wrn_0_5_best.pt"

# SVHN normalization (common values)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

# -----------------------------
# Seed
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Filters / remapping
# -----------------------------
id_set = set(ID_CLASSES)
id_map = {c: i for i, c in enumerate(ID_CLASSES)}

def filter_id(example):
    return int(example["label"]) in id_set

def remap_label(example):
    example["label"] = id_map[int(example["label"])]
    return example

# -----------------------------
# Load SVHN
# HF dataset config: "cropped_digits"
# splits: train / test
# -----------------------------
train_ds = load_dataset("svhn", "cropped_digits", split="train")
test_ds = load_dataset("svhn", "cropped_digits", split="test")

train_id = train_ds.filter(filter_id)
test_id = test_ds.filter(filter_id)

# remap ID labels to 0..NUM_CLASSES-1
train_id = train_id.map(remap_label)
test_id = test_id.map(remap_label)

# train/val split from ID train
splits = train_id.train_test_split(test_size=0.1, seed=SEED)
train_id = splits["train"]
val_id = splits["test"]

print("Train ID size:", len(train_id))
print("Val ID size:", len(val_id))
print("Test ID size:", len(test_id))
print("ID class map:", id_map)

# -----------------------------
# Transforms
# -----------------------------
train_transform = Compose([
    ToTensor(),
    Normalize(SVHN_MEAN, SVHN_STD),
])

eval_transform = Compose([
    ToTensor(),
    Normalize(SVHN_MEAN, SVHN_STD),
])

def apply_train_transforms(examples):
    examples["pixel_values"] = [train_transform(img) for img in examples["image"]]
    return examples

def apply_eval_transforms(examples):
    examples["pixel_values"] = [eval_transform(img) for img in examples["image"]]
    return examples

train_id.set_transform(apply_train_transforms)
val_id.set_transform(apply_eval_transforms)
test_id.set_transform(apply_eval_transforms)

# -----------------------------
# Dataloaders
# -----------------------------
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(
    train_id,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_id,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_id,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# -----------------------------
# WideResNet
# penultimate embedding = global average pooled features
# logits = final fc output
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.shortcut = None if self.equal_in_out else nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)

        shortcut = x if self.equal_in_out else self.shortcut(x)
        return shortcut + out

class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=16, num_classes=6, widen_factor=4, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        n_channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_embedding=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        embedding = out.view(out.size(0), -1)
        logits = self.fc(embedding)

        if return_embedding:
            return logits, embedding
        return logits

model = WideResNet(
    depth=DEPTH,
    num_classes=NUM_CLASSES,
    widen_factor=WIDEN_FACTOR,
    drop_rate=DROPOUT_RATE,
).to(DEVICE)

# -----------------------------
# Optimizer / scheduler / loss
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

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
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total

# -----------------------------
# Train
# -----------------------------
best_val_acc = -1.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, batch in enumerate(train_loader, start=1):
        x = batch["pixel_values"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += y.size(0)

        if step % 100 == 0:
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Train Loss: {running_loss / running_total:.4f} | "
                f"Train Acc: {running_correct / running_total:.4f}"
            )

    scheduler.step()

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc = evaluate(model, val_loader, DEVICE)

    print(
        f"\nEpoch {epoch+1} done | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"Saved best checkpoint to {BEST_CKPT} with val_acc={val_acc:.4f}")

# -----------------------------
# Final test
# -----------------------------
print("Loading best checkpoint...")
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))

test_loss, test_acc = evaluate(model, test_loader, DEVICE)
print(f"ID Test Loss: {test_loss:.4f} | ID Test Acc: {test_acc:.4f}")

# -----------------------------
# Save model package
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "depth": DEPTH,
        "widen_factor": WIDEN_FACTOR,
        "dropout_rate": DROPOUT_RATE,
        "num_classes": NUM_CLASSES,
        "id_classes": ID_CLASSES,
        "mean": SVHN_MEAN,
        "std": SVHN_STD,
    },
    os.path.join(SAVE_DIR, "model.pt"),
)
print(f"Saved model to {os.path.join(SAVE_DIR, 'model.pt')}")

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
  python train_svhn_wrn.py
  '''