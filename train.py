"""
Trainings- und Evaluationsschleife inkl. Zeitmessung pro Epoche und
Gesamtlaufzeit - Grundlage für den Vergleich der Trainingszeit zwischen
den drei Strategien.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time_sec: float


@dataclass
class RunResult:
    mode: str
    train_fraction: float
    num_train_samples: int
    trainable_params: int
    total_params: int
    epochs: List[EpochResult] = field(default_factory=list)
    total_time_sec: float = 0.0

    @property
    def best_val_acc(self) -> float:
        return max((e.val_acc for e in self.epochs), default=0.0)

    @property
    def final_val_acc(self) -> float:
        return self.epochs[-1].val_acc if self.epochs else 0.0


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            # ERGÄNZUNG: non_blocking=True für schnelleren Datentransfer zur GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    train_fraction: float,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> RunResult:
    # ERGÄNZUNG: device-Objekt erstellen
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if verbose:
        print(f"Nutze Gerät: {device}")
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    result = RunResult(
        mode=mode,
        train_fraction=train_fraction,
        num_train_samples=len(train_loader.dataset),
        trainable_params=trainable_params,
        total_params=total_params,
    )

    # Hilfsfunktion für korrekte GPU-Zeitmessung
    def sync_if_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync_if_cuda()
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        sync_if_cuda()
        epoch_start = time.perf_counter()

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        sync_if_cuda()
        epoch_time = time.perf_counter() - epoch_start
        result.epochs.append(EpochResult(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time))

        if verbose:
            print(f"[{mode:17s} | frac={train_fraction:.2f}] Epoche {epoch:2d}/{epochs} "
                  f"- train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  ({epoch_time:.1f}s)")

    sync_if_cuda()
    result.total_time_sec = time.perf_counter() - run_start
    return result

