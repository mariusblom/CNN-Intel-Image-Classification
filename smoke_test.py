"""
Schneller Funktionstest der gesamten Pipeline OHNE echten Bilddatensatz
(verwendet SyntheticImageDataset). Dient nur dazu, Programmierfehler in
Modell/Trainingsschleife aufzudecken - liefert KEINE aussagekräftigen
Ergebnisse! Für die eigentliche Untersuchung: run_intel_experiments.py
mit dem echten Intel-Datensatz verwenden.

Hinweis: "feature_extractor" und "finetune" benötigen Internetzugriff zum
Laden der ImageNet-Gewichte über torchvision und werden hier daher
ausgelassen, sofern kein Internetzugriff verfügbar ist.
"""

import torch
from torch.utils.data import DataLoader

from data_utils import SyntheticImageDataset
from model_factory import build_model
from train import train_model

NUM_CLASSES = 4


def main():
    train_ds = SyntheticImageDataset(num_samples=32, num_classes=NUM_CLASSES, image_size=64)
    val_ds = SyntheticImageDataset(num_samples=16, num_classes=NUM_CLASSES, image_size=64)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = build_model("scratch", num_classes=NUM_CLASSES)
    train_model(model, train_loader, val_loader, mode="scratch", train_fraction=1.0, epochs=2, lr=1e-3)

    print("\nSmoke-Test (scratch) erfolgreich abgeschlossen.")
    print("Für 'feature_extractor'/'finetune' zusätzlich testen (Internet nötig):")
    print('    model = build_model("feature_extractor", num_classes=NUM_CLASSES)')
    print('    train_model(model, train_loader, val_loader, mode="feature_extractor", ')
    print('                train_fraction=1.0, epochs=2, lr=1e-3)')


if __name__ == "__main__":
    main()
