"""
Experiment-Skript speziell für den Intel Image Classification Datensatz.

Setzt die in config.py festgelegten Hyperparameter für die drei Strategien
    c) Training from Scratch
    d) Transfer Learning (Feature Extractor)
    e) Fine-Tuning
um, trainiert je Strategie sowohl über verschiedene Trainingsdatenanteile
(für den Datenmengen-Vergleich) als auch ein finales Modell auf 100% der
Daten (für Lernkurven & Konfusionsmatrix), und erzeugt alle für die
Präsentation relevanten Graphen.

Voraussetzung: Intel-Datensatz von Kaggle heruntergeladen & entpackt
(siehe README.md bzw. Docstring von data_utils.get_intel_dataloaders()).

Beispielaufruf:
    python run_intel_experiments.py --data_root ./intel_data --out_dir ./results_intel
"""

import argparse
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import ALL_CONFIGS, DATASET_INFO
from data_utils import get_intel_dataloaders
from model_factory import build_model
from train import train_model, evaluate

MODES = ["scratch", "feature_extractor", "finetune"]
DATA_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="Wurzelordner des entpackten Intel-Datensatzes (enthält seg_train/, seg_test/)")
    p.add_argument("--fractions", type=float, nargs="+", default=DATA_FRACTIONS)
    p.add_argument("--out_dir", type=str, default="./results_intel")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    final_models = {}  # mode -> (model, test_loader, class_names)  bei fraction=1.0

    shared_batch_size = ALL_CONFIGS["scratch"]["batch_size"]  # identisch für alle drei Strategien

    for fraction in args.fractions:
        # val_loader = Teil von seg_train (für Lernkurven & beste Epoche).
        # test_loader = seg_test, wird unten NACH train_model() genau einmal ausgewertet.
        train_loader, val_loader, test_loader, num_classes, class_names = get_intel_dataloaders(
            args.data_root, batch_size=shared_batch_size, train_fraction=fraction
        )
        print(f"\n=== Trainingsdatenanteil: {fraction:.0%} "
              f"({len(train_loader.dataset)} Trainings- / {len(val_loader.dataset)} Val- / "
              f"{len(test_loader.dataset)} Testbilder, {num_classes} Klassen: {class_names}) ===")

        for mode in MODES:
            cfg = ALL_CONFIGS[mode]
            model = build_model(mode, num_classes=num_classes,
                                 finetune_from=cfg.get("finetune_from", "layer4"))
            result = train_model(
                model, train_loader, val_loader,
                mode=mode, train_fraction=fraction,
                epochs=cfg["epochs"], lr=cfg["lr"], device=args.device,
            )

            # Einmalige, unverfälschte Testauswertung NACH Abschluss des Trainings.
            test_metrics = evaluate(model, test_loader, device=args.device)
            result.test_acc = test_metrics["accuracy"]
            result.test_loss = test_metrics["loss"]
            print(f"  -> Finale Test-Accuracy ({mode}, frac={fraction:.2f}): "
                  f"{result.test_acc:.3f}")

            all_results.append(result)

            if fraction == 1.0:
                final_models[mode] = (model, test_loader, class_names)

    _save_csv(all_results, out_dir / "results.csv")
    _save_hyperparams(out_dir / "hyperparameter.csv")

    _plot_accuracy_vs_fraction(all_results, out_dir / "accuracy_vs_datenmenge.png")
    _plot_time_vs_fraction(all_results, out_dir / "trainingszeit_vs_datenmenge.png")
    _plot_learning_curves(all_results, out_dir / "lernkurven_volle_daten.png")
    _plot_summary_bar(all_results, out_dir / "vergleich_endergebnis.png")
    _plot_trainable_params(all_results, out_dir / "trainierbare_parameter.png")

    for mode, (model, val_loader, class_names) in final_models.items():
        _plot_confusion_matrix(model, val_loader, class_names,
                                out_dir / f"konfusionsmatrix_{mode}.png",
                                mode=mode, device=args.device)

    print(f"\nAlle Ergebnisse und Graphen gespeichert in: {out_dir.resolve()}")


def _save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "train_fraction", "num_train_samples",
                          "trainable_params", "total_params",
                          "best_val_acc", "final_val_acc",
                          "test_acc", "test_loss", "total_time_sec"])
        for r in results:
            writer.writerow([r.mode, r.train_fraction, r.num_train_samples,
                              r.trainable_params, r.total_params,
                              round(r.best_val_acc, 4), round(r.final_val_acc, 4),
                              round(r.test_acc, 4) if r.test_acc is not None else "",
                              round(r.test_loss, 4) if r.test_loss is not None else "",
                              round(r.total_time_sec, 1)])


def _save_hyperparams(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "epochs", "batch_size", "optimizer", "lr",
                          "weight_decay", "frozen_layers", "trainable_layers"])
        for mode, cfg in ALL_CONFIGS.items():
            writer.writerow([mode, cfg["epochs"], cfg["batch_size"], cfg["optimizer"],
                              cfg["lr"], cfg["weight_decay"], cfg.get("frozen_layers", "-"),
                              cfg.get("trainable_layers", "alle")])


def _plot_accuracy_vs_fraction(results, path):
    df = pd.DataFrame([{"mode": r.mode, "fraction": r.train_fraction, "acc": r.test_acc}
                        for r in results])
    plt.figure(figsize=(7, 5))
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("fraction")
        plt.plot(sub["fraction"], sub["acc"], marker="o", label=mode)
    plt.xlabel("Anteil der Trainingsdaten")
    plt.ylabel("Test-Accuracy (seg_test, einmalige Auswertung)")
    plt.title("Prognosegüte vs. Trainingsdatenmenge (Intel Image Classification)")
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        if r.train_fraction == 1.0:
            xs = [e.epoch for e in r.epochs]
            axes[0].plot(xs, [e.val_acc for e in r.epochs], marker="o", label=r.mode)
            axes[1].plot(xs, [e.val_loss for e in r.epochs], marker="o", label=r.mode)
    axes[0].set_xlabel("Epoche"); axes[0].set_ylabel("Val-Accuracy (Split aus seg_train)"); axes[0].set_title("Genauigkeit")
    axes[1].set_xlabel("Epoche"); axes[1].set_ylabel("Val-Loss (Split aus seg_train)"); axes[1].set_title("Loss")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Lernkurven bei 100% der Trainingsdaten (Validation-Split, nicht das Testset)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_summary_bar(results, path):
    df = pd.DataFrame([{"mode": r.mode, "acc": r.test_acc, "time": r.total_time_sec}
                        for r in results if r.train_fraction == 1.0])
    colors = ["#888888", "#4c72b0", "#55a868"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(df["mode"], df["acc"], color=colors)
    axes[0].set_ylabel("Test-Accuracy"); axes[0].set_title("Prognosegüte (100% Daten, finales Testset)")
    axes[1].bar(df["mode"], df["time"], color=colors)
    axes[1].set_ylabel("Trainingszeit (s)"); axes[1].set_title("Trainingszeit (100% Daten)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_trainable_params(results, path):
    df = pd.DataFrame([{"mode": r.mode, "trainable": r.trainable_params, "total": r.total_params}
                        for r in results if r.train_fraction == 1.0]).drop_duplicates("mode")
    plt.figure(figsize=(7, 5))
    x = range(len(df))
    plt.bar(x, df["total"], label="Gesamt (eingefroren + trainierbar)", color="#cccccc")
    plt.bar(x, df["trainable"], label="Trainierbar", color="#4c72b0")
    plt.xticks(list(x), df["mode"])
    plt.ylabel("Anzahl Parameter")
    plt.title("Trainierbare vs. eingefrorene Parameter je Strategie")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion_matrix(model, val_loader, class_names, path, mode, device=None):
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
    plt.title(f"Konfusionsmatrix – {mode}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
