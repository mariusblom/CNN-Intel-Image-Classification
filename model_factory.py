"""
Erzeugt die drei zu vergleichenden Modellvarianten für die
Klassifikationsaufgabe:

  1. "scratch"            - zufällig initialisiert, ALLE Gewichte trainierbar
                             (kein Vortraining, entspricht "Training from Scratch")

  2. "feature_extractor"  - Backbone (conv1 ... layer4) wird mit ImageNet-
                             vortrainierten Gewichten geladen und komplett
                             EINGEFROREN (requires_grad=False). Nur der neue,
                             zufällig initialisierte Klassifikationskopf (fc)
                             wird trainiert.

  3. "finetune"           - Backbone wird ebenfalls mit ImageNet-Gewichten
                             initialisiert, aber nur die SPÄTEN Layer
                             (per Default ab layer4) plus der Kopf werden
                             weiter trainiert (kleine Lernrate empfohlen).
"""

import torch.nn as nn
from resnet18 import ResNet18

STAGE_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]


def load_imagenet_state_dict() -> dict:
    """
    Lädt die offiziellen, auf ImageNet-1k vortrainierten Gewichte über
    torchvision. Da unsere ResNet18-Klasse dieselben Modulnamen verwendet
    wie torchvision.models.resnet18, kann der state_dict 1:1 übernommen
    werden (load_state_dict ohne strict=False nötig).

    Hinweis: Dafür wird beim ersten Aufruf ein Download von
    download.pytorch.org benötigt (Internetzugriff erforderlich).
    """
    from torchvision.models import resnet18, ResNet18_Weights
    tv_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return tv_model.state_dict()


def build_model(mode: str, num_classes: int, finetune_from: str = "layer4") -> nn.Module:
    """
    Parameters
    ----------
    mode : "scratch" | "feature_extractor" | "finetune"
    num_classes : Anzahl der Klassen der neuen Zielaufgabe
    finetune_from : ab welcher Stage beim Fine-Tuning trainiert werden soll
                    (z.B. "layer3" -> layer3, layer4 und fc trainierbar)
    """
    if mode not in {"scratch", "feature_extractor", "finetune"}:
        raise ValueError(f"Unbekannter Modus: {mode}")

    if mode == "scratch":
        model = ResNet18(num_classes=num_classes)
        return model  # Default: alle Parameter trainierbar

    # feature_extractor & finetune: mit vortrainierten Gewichten starten
    model = ResNet18(num_classes=1000)  # 1000 Klassen, wie beim ImageNet-Pretraining
    model.load_state_dict(load_imagenet_state_dict())
    model.fc = nn.Linear(512, num_classes)  # neuer, zufällig initialisierter Kopf

    if mode == "feature_extractor":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc")
        return model

    # mode == "finetune"
    if finetune_from not in STAGE_ORDER:
        raise ValueError(f"finetune_from muss einer von {STAGE_ORDER} sein")
    unfreeze_names = set(STAGE_ORDER[STAGE_ORDER.index(finetune_from):])
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        param.requires_grad = prefix in unfreeze_names
    return model
