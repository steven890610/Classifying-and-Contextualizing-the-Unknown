from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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

NUM_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256

SAVE_DIR = "./mnist_cnn_0_5_model"
BEST_CKPT = "./mnist_cnn_0_5_best.pt"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Filters
# -----------------------------
def filter_id(example):
    return example["label"] < 6   # digits 0-5

def filter_ood(example):
    return example["label"] >= 6  # digits 6-9

# -----------------------------
# Load dataset
# -----------------------------
train_ds, test_ds = load_dataset("mnist", split=["train[:]", "test[:]"])

train_id = train_ds.filter(filter_id)
test_id = test_ds.filter(filter_id)
test_ood = test_ds.filter(filter_ood)

# train/val split from ID train
splits = train_id.train_test_split(test_size=0.1, seed=SEED)
train_id = splits["train"]
val_id = splits["test"]

print("Train ID size:", len(train_id))
print("Val ID size:", len(val_id))
print("Test ID size:", len(test_id))
print("Test OOD size:", len(test_ood))

# -----------------------------
# Transforms
# -----------------------------
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

def apply_transforms(examples):
    examples["pixel_values"] = [transform(image) for image in examples["image"]]
    return examples

train_id.set_transform(apply_transforms)
val_id.set_transform(apply_transforms)
test_id.set_transform(apply_transforms)
test_ood.set_transform(apply_transforms)

# -----------------------------
# Dataloaders
# -----------------------------
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])  # [B,1,28,28]
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

ood_loader = DataLoader(
    test_ood,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# -----------------------------
# CNN model
# embeddings = penultimate layer output
# logits = final classifier output
# -----------------------------
class MNISTCNN(nn.Module):
    def __init__(self, num_classes=6, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 7x7
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, embedding_dim)   # penultimate layer
        self.fc2 = nn.Linear(embedding_dim, num_classes)  # logits layer

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = self.flatten(x)
        embedding = F.relu(self.fc1(x))
        logits = self.fc2(embedding)

        if return_embedding:
            return logits, embedding
        return logits

model = MNISTCNN(num_classes=6, embedding_dim=128).to(DEVICE)

# -----------------------------
# Optimizer / loss
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
# Final ID/OOD checks
# -----------------------------
print("Loading best checkpoint...")
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))

test_loss, test_acc = evaluate(model, test_loader, DEVICE)
print(f"ID Test (0-5) Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

# OOD accuracy is not meaningful because labels 6-9 are outside the 6 trained classes.
# We only run forward passes later for logits/probs/embeddings.

# -----------------------------
# Save full model package
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_classes": 6,
        "embedding_dim": 128,
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
  python train_mnist_cnn.py
  '''