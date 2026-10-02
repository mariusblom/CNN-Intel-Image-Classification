# Transfer Learning mit ResNet-18: Scratch vs. Feature Extractor vs. Fine-Tuning

Untersuchung des Nutzens eines auf ImageNet vortrainierten CNNs für eine neue
Klassifikationsaufgabe (**Intel Image Classification**, 6 Klassen), anhand
einer eigenen, architektonisch identischen Nachbildung von **ResNet-18**
(He et al., 2015).

Datensatz: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

> ⚠️ **Hinweis zur Ausführung:** Der Code wurde in dieser Sandbox aus zwei
> Gründen nicht selbst trainiert: (1) Kaggle ist von hier aus nicht
> erreichbar (Datensatz-Download nicht möglich), (2) es steht kein
> ausreichender Speicherplatz/keine GPU zur Verfügung, um PyTorch zu
> installieren. Bitte lokal oder z. B. in Google Colab ausführen (dort
> stehen Kaggle-Zugriff und GPU zur Verfügung).

---

## 1. Projektstruktur

| Datei | Zweck |
|---|---|
| `resnet18.py` | Eigene ResNet-18-Implementierung (BasicBlock, 4 Stages à 2 Blöcken). Modul-/Parameternamen identisch zu `torchvision.models.resnet18`, damit offizielle ImageNet-Gewichte direkt ladbar sind. ~11,18 Mio. Parameter (6 Klassen). |
| `model_factory.py` | Baut die drei Modellvarianten (`scratch`, `feature_extractor`, `finetune`) inkl. Einfrieren/Freigeben der passenden Layer. |
| `data_utils.py` | Lädt Bilddaten im ImageFolder-Format; `get_intel_dataloaders()` ist speziell auf die Ordnerstruktur des Intel-Datensatzes zugeschnitten; unterstützt Teilmengen der Trainingsdaten (stratifiziert je Klasse). |
| `train.py` | Trainings-/Evaluationsschleife inkl. Zeitmessung je Epoche und insgesamt. |
| `config.py` | **Zentrale Hyperparameter-Tabelle** für die drei Strategien (siehe Abschnitt 3). |
| `run_intel_experiments.py` | Hauptskript: trainiert alle drei Strategien über mehrere Trainingsdatenanteile hinweg und erzeugt alle Graphen für die Auswertung. |
| `run_experiments.py` | Generische Variante für einen beliebigen Datensatz im ImageFolder-Format. |
| `smoke_test.py` | Schneller Funktionstest der Pipeline mit synthetischen Daten (keine echten Ergebnisse, nur Fehlerprüfung). |

---

## 2. Datensatz-Setup

```bash
pip install kaggle
# kaggle.json (API-Token) vorher unter ~/.kaggle/kaggle.json ablegen,
# siehe https://www.kaggle.com/docs/api

kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d intel_data
```

Danach liegt die vom Code erwartete Struktur vor:

```
intel_data/
├── seg_train/seg_train/{buildings,forest,glacier,mountain,sea,street}/*.jpg   (~14.034 Bilder)
├── seg_test/seg_test/{buildings,forest,glacier,mountain,sea,street}/*.jpg     (~3.000 Bilder)
└── seg_pred/seg_pred/*.jpg   (unlabeled, wird hier nicht verwendet)
```

Installation & Ausführung:

```bash
pip install -r requirements.txt
python run_intel_experiments.py --data_root ./intel_data --out_dir ./results_intel
```

---

## 3. Hyperparameter je Trainingsstrategie

### c) Training From Scratch

| Hyperparameter | Wert |
|---|---|
| Epochen | 30 |
| Batch-Size | 64 |
| Optimizer | Adam |
| Lernrate | 1e-3 |
| Weight Decay | 1e-4 |
| LR-Scheduler | StepLR (Faktor 0,5 alle 10 Epochen) |
| Eingefrorene Layer | **keine** |
| Trainierbare Layer | gesamtes Netz (conv1, bn1, layer1–4, fc) — alle ~11,18 Mio. Parameter |
| Initialisierung | zufällig (Kaiming-Init) |

*Begründung:* Ohne ImageNet-Vorwissen muss das Netz alle Merkmale (Kanten, Texturen, Objektteile) selbst erlernen → mehr Epochen und mehr Daten nötig als bei den beiden Transfer-Learning-Varianten.

### d) Transfer Learning (Feature Extractor)

| Hyperparameter | Wert |
|---|---|
| Epochen | 15 |
| Batch-Size | 64 |
| Optimizer | Adam |
| Lernrate | 1e-3 |
| Weight Decay | 1e-4 |
| Eingefrorene Layer | conv1, bn1, layer1, layer2, layer3, layer4 (**gesamtes Backbone**) |
| Trainierbare Layer | nur `fc` (512 → 6 Klassen, ~3.078 Parameter) |
| Initialisierung | Backbone: ImageNet-vortrainiert · `fc`: zufällig |

*Begründung:* Die im Backbone gespeicherten allgemeinen Bildmerkmale werden unverändert übernommen; nur der neue Klassifikationskopf wird trainiert → sehr wenige trainierbare Parameter, kurze Trainingszeit, funktioniert bereits mit wenig Daten.

### e) Fine-Tuning

| Hyperparameter | Wert |
|---|---|
| Epochen | 15 |
| Batch-Size | 64 |
| Optimizer | Adam |
| Lernrate | **1e-4** (kleiner als bei Scratch/Feature Extractor) |
| Weight Decay | 1e-4 |
| Eingefrorene Layer | conv1, bn1, layer1, layer2, layer3 |
| Trainierbare Layer | layer4, fc (~8,4 Mio. Parameter) |
| Initialisierung | Backbone: ImageNet-vortrainiert · `fc`: zufällig |

*Begründung:* Frühe Layer kodieren generische Low-Level-Merkmale und bleiben fix; die abstrakteren, aufgabenspezifischeren späten Layer (`layer4`) werden mit kleiner Lernrate an die Intel-Domäne angepasst (kleine LR, um die guten Startgewichte nicht zu zerstören) → guter Kompromiss zwischen Trainingszeit und Genauigkeit.

Alle drei Konfigurationen sind maschinenlesbar in `config.py` hinterlegt (`ALL_CONFIGS`) und werden von `run_intel_experiments.py` automatisch verwendet; `--finetune_from` in `model_factory.build_model()` erlaubt es, testweise auch ab `layer3` zu entfrieren, falls mehr Anpassung gewünscht ist.

---

## 4. Methodik: Train/Validation/Test-Split

Der Intel-Datensatz liefert von Haus aus nur zwei Ordner (`seg_train`,
`seg_test`). Damit das Testset nicht schon während des Trainings zur
Modellauswahl verwendet wird (Data Leakage), erzeugt `get_intel_dataloaders()`
einen **sauberen Drei-Wege-Split**:

| Split | Herkunft | Verwendung |
|---|---|---|
| **Train** | ~85 % von `seg_train`, stratifiziert je Klasse, mit Augmentierung | Gewichtsupdates |
| **Validation** | ~15 % von `seg_train`, stratifiziert, ohne Augmentierung | Lernkurven, Auswahl der besten Epoche (`best_val_acc`) während des Trainings |
| **Test** | `seg_test`, komplett unangetastet | **Genau einmal** nach Trainingsende ausgewertet (`train.evaluate()`) → die Zahl, die für den Strategievergleich zählt |

Die Trainingsdatenmengen-Experimente (10 %/25 %/50 %/100 %) reduzieren nur
den **Train**-Anteil; Validation und Test bleiben in jedem Lauf vollständig
und unverändert, damit die Vergleichbarkeit über die Datenmengen hinweg
erhalten bleibt.

## 5. Evaluation & erzeugte Graphen

`run_intel_experiments.py` trainiert jede der drei Strategien sowohl bei
**100 % der Trainingsdaten** (für Lernkurven & Konfusionsmatrix) als auch
bei den Anteilen **10 %, 25 %, 50 %, 100 %** (für den Datenmengen-Vergleich)
und speichert in `results_intel/`:

| Datei | Inhalt |
|---|---|
| `results.csv` | Rohdaten je Strategie & Datenanteil: `best_val_acc`/`final_val_acc` (Validation-Split, Trainingsverlauf) sowie **`test_acc`/`test_loss`** (einmalige, finale Auswertung auf `seg_test` – die für den Vergleich maßgebliche Zahl) |
| `hyperparameter.csv` | Die Hyperparameter-Tabellen aus Abschnitt 3 als CSV |
| `accuracy_vs_datenmenge.png` | **Test-Accuracy vs. Trainingsdatenmenge** – zeigt, wie viele Daten jede Strategie für gute Ergebnisse braucht |
| `trainingszeit_vs_datenmenge.png` | **Trainingszeit vs. Trainingsdatenmenge** |
| `lernkurven_volle_daten.png` | Val-Accuracy & Val-Loss über die Epochen bei 100 % der Daten (Trainingsverlauf auf dem Validation-Split, NICHT das Testset) |
| `vergleich_endergebnis.png` | Balkendiagramme: finale Test-Genauigkeit & Trainingszeit im direkten Vergleich |
| `trainierbare_parameter.png` | Trainierbare vs. eingefrorene Parameter je Strategie (visualisiert „Frozen Layers“) |
| `konfusionsmatrix_<mode>.png` | Konfusionsmatrix je Strategie über die 6 Intel-Klassen, berechnet auf dem Testset |

### Erwartete Tendenzen (zur Einordnung der Ergebnisse)

- **Feature Extractor** dürfte bei kleinen Datenmengen (10–25 %) am besten abschneiden und ist mit Abstand am schnellsten trainiert, da nur ~3 Tsd. Parameter aktualisiert werden.
- **Fine-Tuning** benötigt etwas mehr Zeit pro Epoche (mehr trainierbare Parameter), erreicht bei ausreichend Daten (50–100 %) i. d. R. die höchste Endgenauigkeit, da sich `layer4` an die Intel-Bilddomäne (Landschaften/Architektur) anpassen kann.
- **Training from Scratch** benötigt die meisten Epochen/Daten und liegt bei kleinen Datenmengen deutlich zurück; bei 100 % der Daten kann es sich dem Fine-Tuning-Ergebnis annähern, meist aber nicht ganz erreichen.

Diese Tendenzen sind typische Literaturbefunde für Transfer Learning bei
mittelgroßen Datensätzen (~14 Tsd. Bilder) mit moderatem Domain-Shift zu
ImageNet.
