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
    image_size: int = 224,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Lädt den Intel Image Classification Datensatz (Kaggle: puneet6060/intel-image-classification,
    6 Klassen: buildings, forest, glacier, mountain, sea, street; ~14.034 Trainings- und
    ~3.000 Testbilder, Originalauflösung 150x150 Pixel).

    Erwartete Ordnerstruktur nach dem Entpacken des Kaggle-Downloads (Standardlayout des Zips):
        data_root/seg_train/seg_train/<klasse>/*.jpg
        data_root/seg_test/seg_test/<klasse>/*.jpg

    Download (lokal, außerhalb dieser Sandbox - erfordert einen Kaggle-Account/API-Token):
        pip install kaggle
        kaggle datasets download -d puneet6060/intel-image-classification
        unzip intel-image-classification.zip -d data_root

    Returns
    -------
    train_loader, val_loader, num_classes, class_names
    """
    train_dir = Path(data_root) / "seg_train" / "seg_train"
    val_dir = Path(data_root) / "seg_test" / "seg_test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Erwarte '{train_dir}' und '{val_dir}'. Bitte den Intel-Datensatz zuerst von "
            "Kaggle herunterladen und entpacken (siehe Docstring von get_intel_dataloaders)."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(True, image_size))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(False, image_size))
    class_names = train_ds.classes
    num_classes = len(class_names)

    if train_fraction < 1.0:
        train_ds = _stratified_subset(train_ds, train_fraction, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, num_classes, class_names


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
