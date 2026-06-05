# osr_tinyimagenet_vgg32_energy_features_logits_probs.py
#
# Trains a VGG32-style CNN (Option A) on 20 Tiny ImageNet classes (ID only), saves:
#   (1) Penultimate-layer representations (pooled features before FC head)
#   (2) Final-layer logits
#   (3) Final-layer softmax probabilities
# for:
#   - ID train
#   - ID val
#   - OOD val (the other 180 classes)
#
# IMPORTANT: The 180 classes are NEVER used during training or threshold selection.
# Threshold tau is chosen using ID-val scores only.
#
# Data layout required (ImageFolder):
#   <data_root>/
#     train/<synset>/*.JPEG
#     val/<synset>/*.JPEG
#
# Run:
#   python osr_tinyimagenet_vgg32_energy_features_logits_probs.py

import os
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_root: str = "tinyimagenet"
    seed: int = 0

    # If None -> randomly pick 20 synsets from train (deterministic via seed)
    # Else -> provide exactly 20 synset folder names
    id_synsets: Optional[List[str]] = None
    n_id_classes: int = 20

    num_workers: int = 1
    batch_size: int = 128
    epochs: int = 200

    # Adam
    lr: float = 1e-3
    weight_decay: float = 5e-4

    # Energy score / thresholding
    energy_temp: float = 1.0
    target_tpr: float = 0.95  # choose threshold so ~95% of ID-val are accepted

    # Output
    out_dir: str = "outputs_osr"
    save_prefix: str = "vgg32"
    ckpt_path: str = "vgg32_id20_best.pt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_id_classes_from_train(train_ds: ImageFolder, n_id: int, seed: int) -> List[str]:
    classes = list(train_ds.classes)
    rng = random.Random(seed)
    rng.shuffle(classes)
    return classes[:n_id]


def filter_imagefolder_to_classes(ds: ImageFolder, keep_classnames: List[str]) -> Tuple[Subset, Dict[int, int]]:
    """
    Returns:
      subset: Subset(ds) containing only samples in keep_classnames
      old_to_new: mapping old_class_index -> new_class_index in [0..K-1]
                 based on keep_classnames order
    """
    class_to_idx = ds.class_to_idx
    keep_old_idxs = {class_to_idx[c] for c in keep_classnames}
    old_to_new = {class_to_idx[c]: i for i, c in enumerate(keep_classnames)}

    indices = []
    for i, (_, y) in enumerate(ds.samples):
        if y in keep_old_idxs:
            indices.append(i)

    return Subset(ds, indices), old_to_new


def filter_imagefolder_excluding_classes(ds: ImageFolder, exclude_classnames: List[str]) -> Subset:
    exclude_idxs = {ds.class_to_idx[c] for c in exclude_classnames}
    indices = [i for i, (_, y) in enumerate(ds.samples) if y not in exclude_idxs]
    return Subset(ds, indices)


class RemapTargetsDataset(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, old_to_new: Dict[int, int]):
        self.subset = subset
        self.old_to_new = old_to_new

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y_old = self.subset[idx]
        return x, self.old_to_new[y_old]


# ----------------------------
# VGG32-style (Option A) for 64x64
# ----------------------------
def _make_vgg_block(in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    ch = in_ch
    for _ in range(n_convs):
        layers += [
            nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        ch = out_ch
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG32(nn.Module):
    """
    VGG-like CNN adapted for TinyImageNet 64x64, with a clean embedding:
      - 5 blocks, each ends with MaxPool2d(2)
      - global avg pool -> [B, C]
      - linear classifier

    Block conv counts: 2,2,3,3,3 (total conv layers = 13)
    """
    def __init__(self, num_classes: int = 20, width: int = 64, dropout: float = 0.0):
        super().__init__()

        self.features_net = nn.Sequential(
            _make_vgg_block(3,        width,     n_convs=2),   # 64 -> 32
            _make_vgg_block(width,    width*2,   n_convs=2),   # 32 -> 16
            _make_vgg_block(width*2,  width*4,   n_convs=3),   # 16 ->  8
            _make_vgg_block(width*4,  width*8,   n_convs=3),   #  8 ->  4
            _make_vgg_block(width*8,  width*8,   n_convs=3),   #  4 ->  2
        )

        self.feat_dim = width * 8
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.feat_dim, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_net(x)                   # [B, C, 2, 2]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1) # [B, C]
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.fc(feat)
        return logits


# ----------------------------
# Energy scoring + thresholding
# ----------------------------
@torch.no_grad()
def energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    # E(x) = -T * logsumexp(logits/T)
    return -T * torch.logsumexp(logits / T, dim=1)


@torch.no_grad()
def id_score_from_logits(logits: torch.Tensor, T: float) -> torch.Tensor:
    # Higher => more ID-like
    return -energy(logits, T=T)


def threshold_at_tpr(id_scores: List[float], target_tpr: float) -> float:
    # choose tau so that approx target_tpr of scores are >= tau
    s = sorted(id_scores)
    q = 1.0 - target_tpr
    idx = int(math.floor(q * (len(s) - 1)))
    return s[idx]


# ----------------------------
# Train / Eval helpers
# ----------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_ce(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def compute_id_scores(model: nn.Module, loader: DataLoader, device: str, T_energy: float) -> List[float]:
    model.eval()
    scores: List[float] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        score = id_score_from_logits(logits, T=T_energy)
        scores.extend(score.detach().cpu().tolist())
    return scores


# ----------------------------
# Feature + logits/probs extraction
# ----------------------------
@torch.no_grad()
def extract_features_logits_probs(
    model: VGG32,
    loader: DataLoader,
    device: str,
    return_labels: bool = True,
):
    """
    Returns:
      features: [N, D] penultimate pooled features
      logits:   [N, K]
      probs:    [N, K]
      labels:   [N] (only if return_labels)
    """
    model.eval()
    all_feats = []
    all_logits = []
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        feats = model.features(x)
        logits = model.fc(feats)
        probs = torch.softmax(logits, dim=1)

        all_feats.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())

        if return_labels:
            all_labels.append(y.cpu())

    feats = torch.cat(all_feats, dim=0)
    logits = torch.cat(all_logits, dim=0)
    probs = torch.cat(all_probs, dim=0)

    if return_labels:
        labels = torch.cat(all_labels, dim=0)
        return feats, logits, probs, labels

    return feats, logits, probs


# ----------------------------
# Main
# ----------------------------
def main(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # 64x64 transforms
    train_tf = T.Compose([
        T.RandomCrop(64, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        # ImageNet normalization is common even without pretraining
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_full = ImageFolder(os.path.join(cfg.data_root, "train"), transform=train_tf)
    val_full = ImageFolder(os.path.join(cfg.data_root, "val"), transform=eval_tf)

    # Choose ID synsets
    if cfg.id_synsets is None:
        id_synsets = pick_id_classes_from_train(train_full, cfg.n_id_classes, cfg.seed)
    else:
        if len(cfg.id_synsets) != cfg.n_id_classes:
            raise ValueError(f"id_synsets must have length {cfg.n_id_classes}")
        id_synsets = list(cfg.id_synsets)

    # Build ID datasets + remap labels to [0..19]
    train_subset, old2new = filter_imagefolder_to_classes(train_full, id_synsets)
    val_subset, _ = filter_imagefolder_to_classes(val_full, id_synsets)

    train_id = RemapTargetsDataset(train_subset, old2new)
    val_id = RemapTargetsDataset(val_subset, old2new)

    train_loader = DataLoader(
        train_id, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_id, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # OOD val loader (remaining 180 classes) - NEVER used in training/threshold selection.
    ood_val_subset = filter_imagefolder_excluding_classes(val_full, id_synsets)
    ood_val_loader = DataLoader(
        ood_val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # Model + Adam
    model = VGG32(num_classes=cfg.n_id_classes, width=64, dropout=0.0).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, cfg.device)
        va_loss, va_acc = eval_ce(model, val_loader, cfg.device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "id_synsets": id_synsets,
                    "old2new": old2new,
                    "cfg": cfg.__dict__,
                },
                os.path.join(cfg.out_dir, cfg.ckpt_path),
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
                f"best val acc {best_val_acc:.3f}"
            )

    ckpt_file = os.path.join(cfg.out_dir, cfg.ckpt_path)
    print(f"\nSaved best checkpoint to: {ckpt_file}")
    print(f"ID synsets ({cfg.n_id_classes}): {id_synsets}")

    # Load best model
    ckpt = torch.load(ckpt_file, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Threshold from ID-val only
    id_scores = compute_id_scores(model, val_loader, cfg.device, T_energy=cfg.energy_temp)
    tau = threshold_at_tpr(id_scores, cfg.target_tpr)
    accept_rate = sum(s >= tau for s in id_scores) / len(id_scores)

    print("\n--- Energy thresholding (ID-val ONLY) ---")
    print(f"Target TPR: {cfg.target_tpr:.2f}")
    print(f"Achieved accept rate on ID-val: {accept_rate:.3f}")
    print(f"Threshold tau (accept if score >= tau): {tau:.6f}")

    # Extract + save (features, logits, probs)
    print("\nExtracting and saving features/logits/probs...")

    id_train_feats, id_train_logits, id_train_probs, id_train_labels = extract_features_logits_probs(
        model, train_loader, cfg.device, return_labels=True
    )
    id_val_feats, id_val_logits, id_val_probs, id_val_labels = extract_features_logits_probs(
        model, val_loader, cfg.device, return_labels=True
    )
    ood_val_feats, ood_val_logits, ood_val_probs = extract_features_logits_probs(
        model, ood_val_loader, cfg.device, return_labels=False
    )

    # Save outputs with prefix
    p = cfg.save_prefix
    out_id_train = os.path.join(cfg.out_dir, f"{p}_id_train_feats_logits_probs.pt")
    out_id_val = os.path.join(cfg.out_dir, f"{p}_id_val_feats_logits_probs.pt")
    out_ood_val = os.path.join(cfg.out_dir, f"{p}_ood_val_feats_logits_probs.pt")
    out_meta = os.path.join(cfg.out_dir, f"{p}_meta.pt")

    torch.save(
        {
            "features": id_train_feats,     # [N, D]
            "logits": id_train_logits,      # [N, 20]
            "probs": id_train_probs,        # [N, 20]
            "labels": id_train_labels,      # [N]
        },
        out_id_train,
    )
    torch.save(
        {
            "features": id_val_feats,
            "logits": id_val_logits,
            "probs": id_val_probs,
            "labels": id_val_labels,
        },
        out_id_val,
    )
    torch.save(
        {
            "features": ood_val_feats,      # [N_ood, D]
            "logits": ood_val_logits,       # [N_ood, 20] (logits over ID classes)
            "probs": ood_val_probs,         # [N_ood, 20]
        },
        out_ood_val,
    )
    torch.save(
        {
            "id_synsets": id_synsets,
            "n_id_classes": int(cfg.n_id_classes),
            "feat_dim": int(model.feat_dim),
            "energy_temp": float(cfg.energy_temp),
            "target_tpr": float(cfg.target_tpr),
            "threshold_tau": float(tau),
            "checkpoint": ckpt_file,
            "seed": int(cfg.seed),
        },
        out_meta,
    )

    print("\nSaved:")
    print(f"  {out_id_train}")
    print(f"  {out_id_val}")
    print(f"  {out_ood_val}")
    print(f"  {out_meta}")
    print(f"Feature dim: {model.feat_dim}")
    print("\nDone.")


if __name__ == "__main__":
    main(cfg)


'''
apptainer exec --nv \
  --bind /projects/academic/kylehunt/osr_tiny:/projects/academic/kylehunt/osr_tiny \
  --pwd  /projects/academic/kylehunt/osr_tiny \
  "$SIF" \
  python tiny_vgg.py
'''
