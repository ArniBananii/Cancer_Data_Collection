# Dokumentation
## Vorgehen
1. Preprocessing und Daten-Setups stabilisieren
2. Modell-Architektur auswählen
3. Hyperparameter optimieren
4. Finetuning


## 1. Preprocessing und Daten-Setups stabilisieren
### Normalisierung
Normalisierung passend für unseren Datensatz berechnen. Wenn die Normalisiserung ein mal berechnet wurde, muss sie nicht noch mal veränderrt werden. Man rtestet nur, ob es mit Normalisierung besser oder schlechter ist: 

Berechnung der Normalisierung für unsern Datensatz:

Berechne Mean und Std für Normalize()
"""
 Nur ToTensor(), keine Augmentierung oder Normalisierung!
tmp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

tmp_dataset = datasets.ImageFolder(root='Data/Train', transform=tmp_transform)
tmp_loader = DataLoader(tmp_dataset, batch_size=64, shuffle=False, num_workers=2)

mean = 0.
std = 0.
nb_samples = 0.

for data, _ in tmp_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")


Noramlisierungs-Werte für unseren Datensatz

Mean: [0.7438, 0.5865, 0.5869]
Std: [0.0804, 0.1076, 0.1202]

mit Normalisierung:
Epoch [1/10], Loss: 0.974, Acc: 0.650, F1-Score: 0.650: 100%|██████████| 35/35 [11:13<00:00, 19.26s/it]
Epoch [2/10], Loss: 0.874, Acc: 0.685, F1-Score: 0.685: 100%|██████████| 35/35 [11:27<00:00, 19.64s/it]
Epoch [3/10], Loss: 0.734, Acc: 0.735, F1-Score: 0.735: 100%|██████████| 35/35 [11:31<00:00, 19.77s/it]
Epoch [4/10], Loss: 0.675, Acc: 0.749, F1-Score: 0.749: 100%|██████████| 35/35 [11:44<00:00, 20.12s/it]
Epoch [5/10], Loss: 0.653, Acc: 0.763, F1-Score: 0.763: 100%|██████████| 35/35 [12:38<00:00, 21.67s/it]
Epoch [6/10], Loss: 0.639, Acc: 0.760, F1-Score: 0.760: 100%|██████████| 35/35 [11:37<00:00, 19.92s/it]
Epoch [7/10], Loss: 0.589, Acc: 0.786, F1-Score: 0.786: 100%|██████████| 35/35 [13:44<00:00, 23.56s/it]
Epoch [8/10], Loss: 0.528, Acc: 0.807, F1-Score: 0.807: 100%|██████████| 35/35 [12:29<00:00, 21.40s/it]
Epoch [9/10], Loss: 0.505, Acc: 0.808, F1-Score: 0.808: 100%|██████████| 35/35 [12:25<00:00, 21.95s/it]

ohne Normalisierung:
Epoch [1/10], Loss: 2.469, Acc: 0.418, F1-Score: 0.418: 100%|██████████| 35/35 [02:53<00:00,  4.96s/it]
Epoch [2/10], Loss: 1.132, Acc: 0.595, F1-Score: 0.595: 100%|██████████| 35/35 [03:01<00:00,  5.18s/it]
Epoch [3/10], Loss: 0.998, Acc: 0.642, F1-Score: 0.642: 100%|██████████| 35/35 [03:02<00:00,  5.21s/it]
Epoch [4/10], Loss: 0.900, Acc: 0.684, F1-Score: 0.684: 100%|██████████| 35/35 [02:53<00:00,  4.95s/it]
Epoch [5/10], Loss: 0.709, Acc: 0.737, F1-Score: 0.737: 100%|██████████| 35/35 [03:04<00:00,  5.27s/it]
Epoch [6/10], Loss: 0.648, Acc: 0.760, F1-Score: 0.760: 100%|██████████| 35/35 [03:09<00:00,  5.42s/it]
Epoch [7/10], Loss: 0.658, Acc: 0.760, F1-Score: 0.760: 100%|██████████| 35/35 [03:03<00:00,  5.24s/it]
Epoch [8/10], Loss: 0.600, Acc: 0.779, F1-Score: 0.779: 100%|██████████| 35/35 [03:10<00:00,  5.45s/it]
Epoch [9/10], Loss: 0.527, Acc: 0.801, F1-Score: 0.801: 100%|██████████| 35/35 [03:34<00:00,  6.13s/it]

Angebliches Problem mit der Normalisierung und den vortrainierten gewichte der Modelle

Test mit Default Normalisierungswerte und den Default Gewichten
-> ist das Beste Weil deine vortrainierten MobileNet-Gewichte schlicht darauf „getrimmt“ sind, Bilder zu verarbeiten, die mit genau den ImageNet-Statistiken normalisiert wurden. Deine selbst berechneten Mean/Std-Werte reflektieren zwar perfekt die Verteilung deines Datensatzes, passen aber nicht zu den Feature-Filtern und BatchNorm-Parametern, die MobileNet im Pretraining auf den ImageNet-Werten gelernt hat.

  0%|          | 0/35 [00:00<?, ?it/s]Python(91681) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91682) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [1/10], Loss: 2.478, Acc: 0.410, F1-Score: 0.410: 100%|██████████| 35/35 [03:31<00:00,  6.04s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(91782) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91783) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [2/10], Loss: 1.056, Acc: 0.622, F1-Score: 0.622: 100%|██████████| 35/35 [02:38<00:00,  4.54s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(91841) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91842) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [3/10], Loss: 0.936, Acc: 0.659, F1-Score: 0.659: 100%|██████████| 35/35 [03:27<00:00,  5.92s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(91915) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91916) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [4/10], Loss: 0.799, Acc: 0.713, F1-Score: 0.713: 100%|██████████| 35/35 [03:04<00:00,  5.28s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(91949) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91950) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [5/10], Loss: 0.718, Acc: 0.742, F1-Score: 0.742: 100%|██████████| 35/35 [03:17<00:00,  5.65s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(91992) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(91993) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [6/10], Loss: 0.691, Acc: 0.757, F1-Score: 0.757: 100%|██████████| 35/35 [03:07<00:00,  5.35s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(92024) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(92025) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [7/10], Loss: 0.635, Acc: 0.769, F1-Score: 0.769: 100%|██████████| 35/35 [03:22<00:00,  5.79s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(92073) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(92074) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [8/10], Loss: 0.559, Acc: 0.802, F1-Score: 0.802: 100%|██████████| 35/35 [04:22<00:00,  7.50s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(92205) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(92206) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [9/10], Loss: 0.552, Acc: 0.807, F1-Score: 0.807: 100%|██████████| 35/35 [04:16<00:00,  7.32s/it]
  0%|          | 0/35 [00:00<?, ?it/s]Python(92301) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Python(92302) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
Epoch [10/10], Loss: 0.485, Acc: 0.827, F1-Score: 0.827: 100%|██████████| 35/35 [04:03<00:00,  6.95s/it]
Test mit eigener Normalisierungswerte und den Default Gewichten

Epoch [1/15], Loss: 2.531, Acc: 0.402, F1-Score: 0.402: 100%|██████████| 35/35 [03:37<00:00,  6.20s/it]
Epoch [2/15], Loss: 1.103, Acc: 0.614, F1-Score: 0.614: 100%|██████████| 35/35 [03:17<00:00,  5.64s/it]
Epoch [3/15], Loss: 0.976, Acc: 0.655, F1-Score: 0.655: 100%|██████████| 35/35 [03:28<00:00,  5.95s/it]
Epoch [4/15], Loss: 0.844, Acc: 0.695, F1-Score: 0.695: 100%|██████████| 35/35 [03:18<00:00,  5.68s/it]
Epoch [5/15], Loss: 0.743, Acc: 0.733, F1-Score: 0.733: 100%|██████████| 35/35 [03:30<00:00,  6.00s/it]
Epoch [6/15], Loss: 0.728, Acc: 0.744, F1-Score: 0.744: 100%|██████████| 35/35 [03:18<00:00,  5.68s/it]
Epoch [7/15], Loss: 0.672, Acc: 0.754, F1-Score: 0.754: 100%|██████████| 35/35 [03:09<00:00,  5.41s/it]
Epoch [8/15], Loss: 0.569, Acc: 0.795, F1-Score: 0.795: 100%|██████████| 35/35 [03:21<00:00,  5.75s/it]
Epoch [9/15], Loss: 0.561, Acc: 0.807, F1-Score: 0.807: 100%|██████████| 35/35 [02:50<00:00,  4.88s/it]
Epoch [10/15], Loss: 0.551, Acc: 0.799, F1-Score: 0.799: 100%|██████████| 35/35 [03:02<00:00,  5.21s/it]
Epoch [11/15], Loss: 0.500, Acc: 0.828, F1-Score: 0.828: 100%|██████████| 35/35 [03:14<00:00,  5.56s/it]
Epoch [12/15], Loss: 0.470, Acc: 0.837, F1-Score: 0.837: 100%|██████████| 35/35 [03:28<00:00,  5.94s/it]
Epoch [13/15], Loss: 0.471, Acc: 0.836, F1-Score: 0.836: 100%|██████████| 35/35 [03:22<00:00,  5.78s/it]
Epoch [14/15], Loss: 0.452, Acc: 0.850, F1-Score: 0.850: 100%|██████████| 35/35 [03:27<00:00,  5.94s/it]
Epoch [15/15], Loss: 0.451, Acc: 0.842, F1-Score: 0.842: 100%|██████████| 35/35 [03:13<00:00,  5.52s/it]

Test mit errechneten Normalisierung und Ohne Standart gewichte war deutlich schlechte GPT meinte, man muss deutlich mehr Epochen nehmen, damit da überhaipt was vergleichbares rauskommt
 
Epoch [1/10], Loss: 2.622, Acc: 0.146, F1-Score: 0.146: 100%|██████████| 35/35 [03:05<00:00,  5.31s/it]
Epoch [2/10], Loss: 2.041, Acc: 0.213, F1-Score: 0.213: 100%|██████████| 35/35 [03:11<00:00,  5.48s/it]
Epoch [3/10], Loss: 1.988, Acc: 0.241, F1-Score: 0.241: 100%|██████████| 35/35 [03:22<00:00,  5.78s/it]
Epoch [4/10], Loss: 1.914, Acc: 0.269, F1-Score: 0.269: 100%|██████████| 35/35 [03:06<00:00,  5.34s/it]

	1.	Vortrainierte Gewichte wurden auf Images trainiert, die mit genau den ImageNet-Statistiken
0.485, 0.456, 0.406/0.229, 0.224, 0.225 normalisiert wurden.
Die frühen Layer-Filter haben sich an diese Verteilungen „gewöhnt“.
	2.	Eigenen Mean/Std zu verwenden, bedeutet, dass deine Eingabeverteilung („deine“ Bilder) eine ganz andere ist.
Wenn du jetzt mit Dataset-Norm und vortrainierten Gewichten feintunest, kommen die Features nie in den Wertebereich, auf den die Gewichte optimiert wurden – das Fine-Tuning wird instabil oder bleibt schlecht.

Deshalb zwei klare Pfade:
	•	Fine-Tuning → behalte weights=…DEFAULT und nutze immer die zu diesen Gewichten passenden ImageNet-Norm-Werte.
	•	From-Scratch-Training → setze weights=None und verwende deine berechneten Mean/Std, damit die zufälligen Anfangsgewichte vom richtigen Wertebereich ausgehen.


### Augmentation
Testbilder sind immer die gleichen, aber man verändert bzw verzerrt sie mit der Augmentation.

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),

### Welche Zahlen variieren?

| Funktion              | Parameter                          | Empfehlung zum Testen                    |
|-----------------------|-------------------------------------|------------------------------------------|
| RandomHorizontalFlip  | `p` (0.0–1.0)                       | 0.5 ist Standard, 1.0 = immer flip        |
| RandomRotation        | `degrees`                           | 5–20 Grad                                 |
| ColorJitter           | `brightness`, `contrast`, `saturation` | je 0.1–0.4                             |
| RandomResizedCrop     | `scale=(a, b)`                      | (0.7, 1.0) = stärkerer Zoom               |
| RandomPerspective     | `distortion_scale`                  | 0.1–0.5                                   |
|                       | `p`                                 | 0.3–0.7                                   |

        # 1. Horizontale Spiegelung (typisch bei symmetrischen Objekten wie Fahrzeuge, Tiere etc.)
        transforms.RandomHorizontalFlip(p=0.5),  # Wahrscheinlichkeit 50%

        # 2. Leichte Rotation, z.B. bei schiefen Fotos
        transforms.RandomRotation(degrees=10),  # ±10 Grad

        # 3. Farbveränderung: simuliert Beleuchtung
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Werte ~0.1–0.4 sinnvoll

        # 4. Zoom-in + Cropping → robust gegen Bildausschnitte
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 80–100 % Originalgröße

        # 5. Perspektivische Verzerrung (nur wenn sinnvoll)
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # 30 % Verzerrung, 50 % Wahrscheinlichkeit

        transforms.ToTensor(),

        # 6. Normalisierung – hier z. B. ImageNet-Werte oder eure eigenen
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


### Datenbalance

### Wir benutzen Seeding, um das Training des Modells determistisch zu gestalten
Beweis für die Funktion
Epoch [1/15], Loss: 2.518, Acc: 0.400, F1-Score: 0.400: 100%|██████████| 35/35 [02:58<00:00,  5.09s/it]
Epoch [2/15], Loss: 1.075, Acc: 0.617, F1-Score: 0.617: 100%|██████████| 35/35 [03:28<00:00,  5.94s/it]
Epoch [3/15], Loss: 0.954, Acc: 0.655, F1-Score: 0.655: 100%|██████████| 35/35 [03:09<00:00,  4.65s/it]

Epoch [1/15], Loss: 2.518, Acc: 0.400, F1-Score: 0.400: 100%|██████████| 35/35 [03:50<00:00,  6.59s/it]
Epoch [2/15], Loss: 1.075, Acc: 0.617, F1-Score: 0.617: 100%|██████████| 35/35 [03:42<00:00,  6.37s/it]
Epoch [3/15], Loss: 0.954, Acc: 0.655, F1-Score: 0.655: 100%|██████████| 35/35 [05:00<00:00,  8.28s/it]
#### 1. Klassengewichteter Loss (CrossEntropyLoss mit `weight=`)

Hierbei wird dem Verlustwert jeder Klasse ein Gewicht zugewiesen, das invers proportional zur Häufigkeit der Klasse im Trainingsdatensatz ist. Je seltener eine Klasse, desto stärker wird sie im Training gewichtet.

**Beispiel:**  
Wenn Klasse A nur 50 Bilder hat und Klasse B 500, bekommt A ein 10-fach höheres Gewicht.

 Vorteil:
- Einfach zu implementieren
- Muss nicht in den DataLoader eingreifen

 Visualisierung:  
Wir haben die resultierenden Gewichte pro Klasse als Balkendiagramm dargestellt, um sicherzustellen, dass die Gewichtung korrekt berechnet wurde.

---

#### 2. WeightedRandomSampler im DataLoader

Statt den Verlust zu gewichten, können wir das Training direkt auf Batch-Ebene ausbalancieren. Hierzu haben wir den `WeightedRandomSampler` eingesetzt. Er sorgt dafür, dass jede Klasse mit gleicher Wahrscheinlichkeit in einem Trainings-Batch vertreten ist – unabhängig von ihrer Häufigkeit im Datensatz.

Vorteil:
- Balanced Batches → konsistenteres Gradientenverhalten
- Kein Eingriff in die Verlustfunktion notwendig

Evaluation:  
Ein Plot der Klassenverteilung im ersten Trainings-Batch zeigte, dass der Sampler erfolgreich eine annähernd gleichmäßige Verteilung erzeugt.

---

### Fazit:

Beide Methoden haben ihre Stärken. Während der gewichtete Loss mathematisch elegant ist, sorgt der Sampler direkt auf Datenebene für Ausgleich. Für unsere finale Modellpipeline entschieden wir uns für den 
**WeightedRandomSampler**, da er bessere Klassenergebnisse bei den unterrepräsentierten Klassen lieferte.


## 2. Modell-Architektur auswählen
EfficentNet-B0
MobileNetV3
Resnet18, ResNet34 Resnet50
VGG16

## 3. Hyperparameter optimieren
Epochen
Batchsize
Lernrate
Optimizer

## 4. Finetuning
Learning Rate Scheduler (ReduceLROnPlateau, CosineAnnealing, StepLR)
Dropout-Level verändern
ggf. Weight Decay
ggf. Early Stopping
