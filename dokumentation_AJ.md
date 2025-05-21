# Dokumentation
## Vorgehen
1. Preprocessing und Daten-Setups stabilisieren
2. Modell-Architektur auswählen
3. Hyperparameter optimieren
4. Finetuning


## 1. Preprocessing und Daten-Setups stabilisieren
### Normalisierung
Normalisierung passend für unseren Datensatz berechnen. Wenn die Normalisiserung ein mal berechnet wurde, muss sie nicht noch mal veränderrt werden. Man rtestet nur, ob es mit Normalisierung besser oder schlechter ist: 

Vorschlag:

#%%
"""
Berechne Mean und Std für Normalize()
"""
# Nur ToTensor(), keine Augmentierung oder Normalisierung!
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


Noramlisierungs Datensatz für unseren Datensatz

Nutze diese Werte für Normalize():
Mean: [0.7438, 0.5865, 0.5869]
Std: [0.0804, 0.1076, 0.1202]

mit Normalisierung:
Epoch [1/10], Loss: 1.101, Acc: 0.619, F1-Score: 0.619: 100%|██████████| 35/35 [08:55<00:00, 15.30s/it]
Epoch [2/10], Loss: 0.919, Acc: 0.665, F1-Score: 0.665: 100%|██████████| 35/35 [10:16<00:00, 17.63s/it]
Epoch [3/10], Loss: 0.827, Acc: 0.698, F1-Score: 0.698: 100%|██████████| 35/35 [09:42<00:00, 16.64s/it]
Epoch [4/10], Loss: 0.753, Acc: 0.728, F1-Score: 0.728: 100%|██████████| 35/35 [09:41<00:00, 16.61s/it]
Epoch [5/10], Loss: 0.708, Acc: 0.744, F1-Score: 0.744: 100%|██████████| 35/35 [09:46<00:00, 16.75s/it]
Epoch [6/10], Loss: 0.663, Acc: 0.750, F1-Score: 0.750: 100%|██████████| 35/35 [10:51<00:00, 18.62s/it]
Epoch [7/10], Loss: 0.571, Acc: 0.778, F1-Score: 0.778: 100%|██████████| 35/35 [10:52<00:00, 18.63s/it]
Epoch [8/10], Loss: 0.577, Acc: 0.782, F1-Score: 0.782: 100%|██████████| 35/35 [10:53<00:00, 18.68s/it]
Epoch [9/10], Loss: 0.500, Acc: 0.812, F1-Score: 0.812: 100%|██████████| 35/35 [10:29<00:00, 17.97s/it]
Epoch [10/10], Loss: 0.514, Acc: 0.795, F1-Score: 0.795: 100%|██████████| 35/35 [11:05<00:00, 19.01s/it]

ohne Normalisierung:


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

Durch unseren ungleichverteilten Datensatz müssen wir darauf achten, dass wir die Klassen gleichmäßig gewichten – sonst würde das Modell vor allem die häufigen Klassen bevorzugen. Dazu haben wir zwei Ansätze experimentell getestet:

---

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


## Optimizer und Learning rate - Quick Test
##### 🔧 Starte Training mit: SGD_0.01
SGD_0.01 | Loss: 1.978 | Acc: 0.289 | F1: 0.289: 100%|██████████| 35/35 [03:31<00:00,  6.04s/it]
##### SGD_0.01 Epoch 2/2:   0%|          | 0/35 [00:00<?, ?it/s]
SGD_0.01 | Loss: 1.439 | Acc: 0.482 | F1: 0.482: 100%|██████████| 35/35 [03:00<00:00,  5.15s/it]
##### ✅ SGD_0.01 → Test Accuracy: 50.000%


##### 🔧 Starte Training mit: SGD_0.001
SGD_0.001 | Loss: 2.199 | Acc: 0.127 | F1: 0.127: 100%|██████████| 35/35 [02:42<00:00,  4.65s/it]
SGD_0.001 Epoch 2/2:   0%|          | 0/35 [00:00<?, ?it/s]
SGD_0.001 | Loss: 2.112 | Acc: 0.230 | F1: 0.230: 100%|██████████| 35/35 [02:47<00:00,  4.79s/it]
##### ✅ SGD_0.001 → Test Accuracy: 34.746%


##### 🔧 Starte Training mit: Adam_0.001
Adam_0.001 | Loss: 1.537 | Acc: 0.444 | F1: 0.444: 100%|██████████| 35/35 [02:48<00:00,  4.81s/it]
Adam_0.001 Epoch 2/2:   0%|          | 0/35 [00:00<?, ?it/s]
Adam_0.001 | Loss: 1.115 | Acc: 0.588 | F1: 0.588: 100%|██████████| 35/35 [02:50<00:00,  4.87s/it]
##### ✅ Adam_0.001 → Test Accuracy: 52.542%


##### 🔧 Starte Training mit: Adam_0.0001
Adam_0.0001 | Loss: 2.114 | Acc: 0.229 | F1: 0.229: 100%|██████████| 35/35 [02:51<00:00,  4.90s/it]
Adam_0.0001 Epoch 2/2:   0%|          | 0/35 [00:00<?, ?it/s]
Adam_0.0001 | Loss: 1.827 | Acc: 0.431 | F1: 0.431: 100%|██████████| 35/35 [02:40<00:00,  4.60s/it]
##### ✅ Adam_0.0001 → Test Accuracy: 50.000%

##### 🔧 Starte Training mit: RMSprop_0.001
RMSprop_0.001 | Loss: 1.930 | Acc: 0.295 | F1: 0.295: 100%|██████████| 35/35 [02:40<00:00,  4.59s/it]
RMSprop_0.001 Epoch 2/2:   0%|          | 0/35 [00:00<?, ?it/s]
RMSprop_0.001 | Loss: 1.354 | Acc: 0.510 | F1: 0.510: 100%|██████████| 35/35 [02:50<00:00,  4.87s/it]
##### ✅ RMSprop_0.001 → Test Accuracy: 51.695%

### 📊 Optimizer-Vergleich

| Optimizer       | Accuracy |
|-----------------|--------|
| Adam\_0.001     | 52.54% |
| RMSprop\_0.001  | 51.69% |
| SGD\_0.01       | 50.00% |
| Adam\_0.0001    | 50.00% |
| SGD\_0.001      | 34.75% |



## 2. Modell-Architektur auswählen
EfficentNet-B0
MobileNetV3
Resnet18, ResNet34 Resnet50
VGG16

## 3. Hyperparameter optimieren
Epochen
Batchsize
Lernrate
Optimizer?
Adam?

## 4. Finetuning
Learning Rate Scheduler (ReduceLROnPlateau, CosineAnnealing, StepLR)
Dropout-Level verändern
ggf. Weight Decay
ggf. Early Stopping


AJ:
Learning Scheduler
Optimizer - 5 oder 10 Epochen!!
