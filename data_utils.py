"""
Daten-Utilities:
  - get_dataloaders(): lädt einen Bilddatensatz im ImageFolder-Format
        data_dir/train/<klasse>/*.jpg
        data_dir/val/<klasse>/*.jpg
    und erlaubt, nur einen Bruchteil der Trainingsdaten zu verwenden
    (stratifiziert je Klasse) - zentral für die Untersuchung "benötigte
    Menge an Trainingsdaten".

  - SyntheticImageDataset: rein synthetischer Ersatzdatensatz, NUR für
    den Funktionstest der Pipeline ohne echten Datensatz (smoke_test.py).
"""

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    train_fraction: float = 1.0,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(True, image_size))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(False, image_size))
    num_classes = len(train_ds.classes)

    if train_fraction < 1.0:
        train_ds = _stratified_subset(train_ds, train_fraction, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, num_classes


def _stratified_subset(dataset: datasets.ImageFolder, fraction: float, seed: int) -> Subset:
    """Zieht je Klasse denselben Anteil `fraction` der Trainingsbeispiele."""
    rng = random.Random(seed)
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class.setdefault(label, []).append(idx)

    selected = []
    for indices in indices_by_class.values():
        rng.shuffle(indices)
        k = max(1, int(round(len(indices) * fraction)))
        selected.extend(indices[:k])

    return Subset(dataset, selected)


def get_intel_dataloaders(
    data_root: str,
    batch_size: int = 64,
    train_fraction: float = 1.0,
    val_fraction: float = 0.15,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Lädt den Intel Image Classification Datensatz (Kaggle: puneet6060/intel-image-classification,
    6 Klassen: buildings, forest, glacier, mountain, sea, street; ~14.034 Trainings- und
    ~3.000 Testbilder, Originalauflösung 150x150 Pixel) mit einem SAUBEREN
    Drei-Wege-Split:

        - train_loader: Teilmenge von seg_train, MIT Augmentierung. Wird für
          die Gewichtsupdates verwendet.
        - val_loader:   anderer, disjunkter Teil von seg_train (Anteil
          `val_fraction`, stratifiziert je Klasse gezogen), OHNE
          Augmentierung. Dient während des Trainings der Verlaufsbeobachtung
          (Lernkurven) und der Auswahl der besten Epoche (best_val_acc).
        - test_loader:  seg_test, komplett unangetastet. Wird NICHT während
          des Trainings verwendet, sondern NUR EIN EINZIGES MAL danach
          ausgewertet, um eine unverfälschte, finale Testgenauigkeit zu
          erhalten (siehe train.evaluate()).

    Dadurch wird vermieden, den eigentlichen Testdatensatz schon während des
    Trainings zur Modellauswahl heranzuziehen (Data Leakage).

    Erwartete Ordnerstruktur nach dem Entpacken des Kaggle-Downloads (Standardlayout des Zips):
        data_root/seg_train/seg_train/<klasse>/*.jpg
        data_root/seg_test/seg_test/<klasse>/*.jpg

    Download (lokal, außerhalb dieser Sandbox - erfordert einen Kaggle-Account/API-Token):
        pip install kaggle
        kaggle datasets download -d puneet6060/intel-image-classification
        unzip intel-image-classification.zip -d data_root

    Returns
    -------
    train_loader, val_loader, test_loader, num_classes, class_names
    """
    train_dir = Path(data_root) / "seg_train" / "seg_train"
    test_dir = Path(data_root) / "seg_test" / "seg_test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Erwarte '{train_dir}' und '{test_dir}'. Bitte den Intel-Datensatz zuerst von "
            "Kaggle herunterladen und entpacken (siehe Docstring von get_intel_dataloaders)."
        )

    # Zwei ImageFolder-Instanzen auf demselben Ordner mit unterschiedlichen
    # Transforms (Training braucht Augmentierung, Validierung nicht). Die
    # Sample-Reihenfolge ist bei beiden identisch -> dieselben Indizes passen
    # auf beide Instanzen.
    train_ds_aug = datasets.ImageFolder(train_dir, transform=build_transforms(True, image_size))
    train_ds_eval = datasets.ImageFolder(train_dir, transform=build_transforms(False, image_size))
    class_names = train_ds_aug.classes
    num_classes = len(class_names)

    train_idx, val_idx = _stratified_train_val_split(train_ds_aug, val_fraction, seed)

    if train_fraction < 1.0:
        train_idx = _stratified_index_fraction(train_ds_aug, train_idx, train_fraction, seed)

    train_subset = Subset(train_ds_aug, train_idx)
    val_subset = Subset(train_ds_eval, val_idx)
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(False, image_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, num_classes, class_names


def _stratified_train_val_split(dataset: datasets.ImageFolder, val_fraction: float, seed: int):
    """Teilt die Indizes von `dataset` je Klasse in zwei disjunkte Mengen (train, val)."""
    rng = random.Random(seed)
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for indices in indices_by_class.values():
        indices = list(indices)
        rng.shuffle(indices)
        k_val = max(1, int(round(len(indices) * val_fraction)))
        val_idx.extend(indices[:k_val])
        train_idx.extend(indices[k_val:])
    return train_idx, val_idx


def _stratified_index_fraction(dataset: datasets.ImageFolder, indices, fraction: float, seed: int):
    """Zieht `fraction` der gegebenen Indizes, stratifiziert je Klasse (für die
    Trainingsdatenmengen-Experimente; betrifft NUR den Trainingsanteil, nie val/test)."""
    rng = random.Random(seed + 1)
    indices_by_class = {}
    for idx in indices:
        label = dataset.samples[idx][1]
        indices_by_class.setdefault(label, []).append(idx)

    selected = []
    for class_indices in indices_by_class.values():
        class_indices = list(class_indices)
        rng.shuffle(class_indices)
        k = max(1, int(round(len(class_indices) * fraction)))
        selected.extend(class_indices[:k])
    return selected


class SyntheticImageDataset(Dataset):
    """
    Rein synthetischer Ersatzdatensatz (zufällige Tensoren mit leichtem
    klassenabhängigem Signal). Dient AUSSCHLIESSLICH dazu, die
    Trainings-Pipeline ohne echten Bilddatensatz auf Programmierfehler zu
    prüfen. Liefert KEINE aussagekräftigen Klassifikationsergebnisse -
    für die eigentliche Untersuchung bitte einen echten Datensatz im
    ImageFolder-Format über get_dataloaders() verwenden.
    """

    def __init__(self, num_samples: int, num_classes: int, image_size: int = 64):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = idx % self.num_classes
        g = torch.Generator().manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size, generator=g) * 0.1
        image[label % 3] += 0.5  # leichtes, lernbares Klassensignal
        return image, label
