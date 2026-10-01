"""
Hauptskript: führt alle drei Trainingsstrategien (scratch,
feature_extractor, finetune) über mehrere Trainingsdatenmengen hinweg
aus und speichert Ergebnisse als CSV sowie Vergleichsdiagramme.

Voraussetzung: Bilddatensatz im ImageFolder-Format
    data_dir/train/<klasse_1>/*.jpg
    data_dir/train/<klasse_2>/*.jpg
    ...
    data_dir/val/<klasse_1>/*.jpg
    ...

Beispielaufruf:
    python run_experiments.py --data_dir ./data --epochs 10 \
        --fractions 0.1 0.25 0.5 1.0
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import get_dataloaders
from model_factory import build_model
from train import train_model

import torch

MODES = ["scratch", "feature_extractor", "finetune"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                    help="Ordner mit data_dir/train/<klasse>/... und data_dir/val/<klasse>/...")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0],
                    help="Zu testende Anteile der Trainingsdaten")
    p.add_argument("--finetune_from", type=str, default="layer4",
                    help="Ab welcher Stage beim Fine-Tuning trainiert wird")
    p.add_argument("--out_dir", type=str, default="./results")
    return p.parse_args()



def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # <--- NEU: Device bestimmen (CUDA / GPU falls vorhanden)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwendetes Device: {device}")

    all_results = []

    for fraction in args.fractions:
        train_loader, val_loader, num_classes = get_dataloaders(
            args.data_dir, batch_size=args.batch_size, train_fraction=fraction
        )
        print(f"\n=== Trainingsdatenanteil: {fraction:.0%}  "
              f"({len(train_loader.dataset)} Bilder, {num_classes} Klassen) ===")

        for mode in MODES:
            model = build_model(mode, num_classes=num_classes, finetune_from=args.finetune_from)
            
            # <--- NEU: Modell auf die GPU verschieben
            model = model.to(device)

            result = train_model(
                model, train_loader, val_loader,
                mode=mode, train_fraction=fraction,
                epochs=args.epochs, lr=args.lr,
                device=device  # <--- NEU: Device an train_model übergeben
            )
            all_results.append(result)


def _save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "train_fraction", "num_train_samples",
                          "trainable_params", "total_params",
                          "best_val_acc", "final_val_acc", "total_time_sec"])
        for r in results:
            writer.writerow([r.mode, r.train_fraction, r.num_train_samples,
                              r.trainable_params, r.total_params,
                              round(r.best_val_acc, 4), round(r.final_val_acc, 4),
                              round(r.total_time_sec, 1)])


def _plot_accuracy_vs_fraction(results, path):
    df = pd.DataFrame([{"mode": r.mode, "fraction": r.train_fraction, "acc": r.best_val_acc}
                        for r in results])
    plt.figure(figsize=(7, 5))
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("fraction")
        plt.plot(sub["fraction"], sub["acc"], marker="o", label=mode)
    plt.xlabel("Anteil der Trainingsdaten")
    plt.ylabel("Beste Validierungsgenauigkeit")
    plt.title("Prognosegüte vs. Trainingsdatenmenge")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_time_vs_fraction(results, path):
    df = pd.DataFrame([{"mode": r.mode, "fraction": r.train_fraction, "time": r.total_time_sec}
                        for r in results])
    plt.figure(figsize=(7, 5))
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("fraction")
        plt.plot(sub["fraction"], sub["time"], marker="o", label=mode)
    plt.xlabel("Anteil der Trainingsdaten")
    plt.ylabel("Trainingszeit (Sekunden)")
    plt.title("Trainingszeit vs. Trainingsdatenmenge")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_learning_curves(results, path):
    plt.figure(figsize=(7, 5))
    for r in results:
        if r.train_fraction == max(x.train_fraction for x in results):
            xs = [e.epoch for e in r.epochs]
            ys = [e.val_acc for e in r.epochs]
            plt.plot(xs, ys, marker="o", label=r.mode)
    plt.xlabel("Epoche")
    plt.ylabel("Validierungsgenauigkeit")
    plt.title("Lernkurven bei maximalem Trainingsdatenanteil")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
