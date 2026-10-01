"""
Vergleicht AUSSCHLIESSLICH die Fine-Tuning-Strategie bei unterschiedlicher
Freeze-Tiefe (ab welcher Stage wird entfroren) auf dem Intel-Datensatz:

    finetune_from="layer2"  -> layer2, layer3, layer4, fc trainierbar
    finetune_from="layer3"  -> layer3, layer4, fc trainierbar
    finetune_from="layer4"  -> layer4, fc trainierbar (Standard-Konfiguration)

"scratch" und "feature_extractor" werden hier NICHT ausgeführt.

Alle übrigen Hyperparameter (Epochen, Batch-Size, Lernrate, Optimizer) werden
unverändert aus config.FINETUNE übernommen, nur finetune_from wird variiert.

Beispielaufruf:
    python run_finetune_layer_comparison.py --data_root ./intel_data --variants layer2 layer3 layer4
"""

import argparse
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import FINETUNE, DATASET_INFO
from data_utils import get_intel_dataloaders
from model_factory import build_model
from train import train_model

FINETUNE_VARIANTS = ["layer2", "layer3", "layer4"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="Wurzelordner des entpackten Intel-Datensatzes (enthält seg_train/, seg_test/)")
    p.add_argument("--variants", type=str, nargs="+", default=FINETUNE_VARIANTS,
                    help="Zu vergleichende finetune_from-Werte, z.B. layer2 layer3 layer4")
    p.add_argument("--out_dir", type=str, default="./results_finetune_layers")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes, class_names = get_intel_dataloaders(
        args.data_root, batch_size=FINETUNE["batch_size"], train_fraction=1.0
    )
    print(f"Datensatz: {len(train_loader.dataset)} Trainingsbilder, "
          f"{num_classes} Klassen: {class_names}")

    all_results = []
    trained_models = {}  # variant -> model (für Konfusionsmatrizen)

    for variant in args.variants:
        print(f"\n=== Fine-Tuning ab {variant} ===")
        model = build_model("finetune", num_classes=num_classes, finetune_from=variant)
        result = train_model(
            model, train_loader, val_loader,
            mode=f"finetune_from_{variant}", train_fraction=1.0,
            epochs=FINETUNE["epochs"], lr=FINETUNE["lr"], device=args.device,
        )
        all_results.append((variant, result))
        trained_models[variant] = model

    _save_csv(all_results, out_dir / "finetune_layer_comparison.csv")
    _plot_summary_bar(all_results, out_dir / "finetune_layer_vergleich.png")
    _plot_trainable_params(all_results, out_dir / "finetune_layer_parameter.png")
    _plot_learning_curves(all_results, out_dir / "finetune_layer_lernkurven.png")

    for variant, model in trained_models.items():
        _plot_confusion_matrix(model, val_loader, class_names,
                                out_dir / f"finetune_layer_konfusionsmatrix_{variant}.png",
                                variant, device=args.device)

    print(f"\nErgebnisse gespeichert in: {out_dir.resolve()}")


def _save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["finetune_from", "trainable_params", "total_params",
                          "best_val_acc", "final_val_acc", "total_time_sec"])
        for variant, r in results:
            writer.writerow([variant, r.trainable_params, r.total_params,
                              round(r.best_val_acc, 4), round(r.final_val_acc, 4),
                              round(r.total_time_sec, 1)])


def _plot_summary_bar(results, path):
    variants = [v for v, _ in results]
    accs = [r.best_val_acc for _, r in results]
    times = [r.total_time_sec for _, r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(variants, accs, color="#4c72b0")
    axes[0].set_ylabel("Beste Test-Accuracy"); axes[0].set_title("Genauigkeit je Freeze-Tiefe")
    axes[1].bar(variants, times, color="#55a868")
    axes[1].set_ylabel("Trainingszeit (s)"); axes[1].set_title("Trainingszeit je Freeze-Tiefe")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Fine-Tuning: Vergleich der Freeze-Tiefe (layer2 vs. layer3 vs. layer4)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_trainable_params(results, path):
    variants = [v for v, _ in results]
    trainable = [r.trainable_params for _, r in results]
    total = [r.total_params for _, r in results]

    plt.figure(figsize=(7, 5))
    x = range(len(variants))
    plt.bar(x, total, label="Gesamt", color="#cccccc")
    plt.bar(x, trainable, label="Trainierbar", color="#4c72b0")
    plt.xticks(list(x), [f"ab {v}" for v in variants])
    plt.ylabel("Anzahl Parameter")
    plt.title("Trainierbare Parameter je Freeze-Tiefe")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_learning_curves(results, path):
    plt.figure(figsize=(7, 5))
    for variant, r in results:
        xs = [e.epoch for e in r.epochs]
        ys = [e.val_acc for e in r.epochs]
        plt.plot(xs, ys, marker="o", label=f"ab {variant}")
    plt.xlabel("Epoche")
    plt.ylabel("Test-Accuracy")
    plt.title("Lernkurven: Fine-Tuning bei unterschiedlicher Freeze-Tiefe")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion_matrix(model, val_loader, class_names, path, variant, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Vorhergesagt"); plt.ylabel("Tatsächlich")
    plt.title(f"Konfusionsmatrix – Fine-Tuning ab {variant}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
