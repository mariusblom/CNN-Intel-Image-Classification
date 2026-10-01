"""
Zentrale Konfiguration der Hyperparameter für die drei Trainingsstrategien
auf dem Intel Image Classification Datensatz.

Bildet direkt die Gliederungspunkte der Präsentation ab:
    c) Training From Scratch       -> SCRATCH
    d) Transfer Learning            -> FEATURE_EXTRACTOR
    e) Fine-Tuning                  -> FINETUNE
"""

DATASET_INFO = {
    "name": "Intel Image Classification",
    "source": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    "num_classes": 6,
    "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
    "num_train_images": 14034,
    "num_test_images": 3000,
    "native_image_size": 150,
}

# ---------------------------------------------------------------------------
# c) Training From Scratch
# ---------------------------------------------------------------------------
SCRATCH = {
    "mode": "scratch",
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "StepLR(step_size=10, gamma=0.5)",
    "frozen_layers": "keine – alle ~11,18 Mio. Parameter (6 Klassen) trainierbar",
    "trainable_layers": "gesamtes Netz (conv1, bn1, layer1-4, fc)",
    "gewichtsinitialisierung": "zufällig (Kaiming-Init)",
    "begruendung": (
        "Ohne Vorwissen aus ImageNet muss das Netz alle Merkmale von Grund auf "
        "lernen -> mehr Epochen und tendenziell mehr Trainingsdaten nötig als "
        "bei den beiden folgenden Varianten."
    ),
}

# ---------------------------------------------------------------------------
# d) Transfer Learning (Feature Extractor, Backbone vollständig eingefroren)
# ---------------------------------------------------------------------------
FEATURE_EXTRACTOR = {
    "mode": "feature_extractor",
    "epochs": 15,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "keiner (kurze Trainingsdauer)",
    "frozen_layers": "conv1, bn1, layer1, layer2, layer3, layer4 (gesamtes Backbone)",
    "trainable_layers": "nur fc (neuer Klassifikationskopf, 512 -> 6 Klassen)",
    "gewichtsinitialisierung": "ImageNet-vortrainiert (Backbone), fc zufällig",
    "begruendung": (
        "Das Backbone liefert bereits allgemeine Bildmerkmale (Kanten, Texturen, "
        "Formen); nur der Kopf wird auf die 6 Intel-Klassen angepasst -> sehr "
        "wenige trainierbare Parameter (~3.078), kurze Trainingszeit, wenig "
        "Daten nötig."
    ),
}

# ---------------------------------------------------------------------------
# e) Fine-Tuning (Backbone teilweise entfroren)
# ---------------------------------------------------------------------------
FINETUNE = {
    "mode": "finetune",
    "epochs": 15,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 1e-4,  # kleiner als bei Scratch/Feature-Extractor: bereits gute Startgewichte
    "weight_decay": 1e-4,
    "lr_scheduler": "keiner (kurze Trainingsdauer)",
    "finetune_from": "layer4",
    "frozen_layers": "conv1, bn1, layer1, layer2, layer3",
    "trainable_layers": "layer4, fc (~8,4 Mio. Parameter)",
    "gewichtsinitialisierung": "ImageNet-vortrainiert (Backbone), fc zufällig",
    "begruendung": (
        "Frühe Layer erkennen generische Low-Level-Merkmale (Kanten, Farben) und "
        "bleiben eingefroren; die späten Layer (layer4) codieren abstraktere, "
        "aufgabenspezifischere Merkmale und werden mit kleiner Lernrate an die "
        "Intel-Domäne angepasst -> guter Kompromiss aus Trainingszeit und Genauigkeit."
    ),
}

ALL_CONFIGS = {"scratch": SCRATCH, "feature_extractor": FEATURE_EXTRACTOR, "finetune": FINETUNE}


def print_summary():
    for name, cfg in ALL_CONFIGS.items():
        print(f"\n--- {name} ---")
        for k, v in cfg.items():
            if k != "mode":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    print_summary()
