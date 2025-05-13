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


### Datenbalance?
Durch unseren ungleichverteilten Datensatz müssen wir darauf achten, dass wir die Klassen gleichmäßig verteilen. Das geht mit dem WeightedRandomSampler oder der 


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
