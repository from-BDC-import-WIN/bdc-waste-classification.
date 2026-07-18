# Cara Modifikasi `03_convnext.ipynb` → Swin Tiny

Target: `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`

Base: `notebooks/experiments/03_convnext.ipynb` (copy jadi `04_swin.ipynb`, jangan edit file convnext-nya).

## ⚠️ Perbedaan arsitektur penting

Swin (timm) dengan `global_pool=""` return output **channel-last**: `(B, H, W, C)`, bukan `(B, C, H, W)` seperti ResNet/ConvNeXt. Dicek langsung:

```python
m = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', num_classes=0, global_pool='')
m(torch.randn(2,3,224,224)).shape
# -> torch.Size([2, 7, 7, 768])   <- (B, H, W, C)
```

Jadi wajib `permute(0, 3, 1, 2)` sebelum masuk `nn.Conv2d` di custom head. Kalau lupa, `Conv2d` bakal error shape mismatch (channel dianggap 7, bukan 768).

## Bagian yang perlu diubah

### 1. Cell `EXPERIMENT_CONFIG`
```python
"model": {
    "backbone": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "fine_tune_at": 140,
},
```
`fine_tune_at=140` dicek dari `named_parameters()` backbone (total 171 tensor). Stage terakhir (`layers.3.*`) mulai idx 140, sampai final `norm` di idx 169-170 → freeze index `[:140]`, unfreeze sisanya (31 tensor) di Phase 2. Analog ke pola `fine_tune_at=162` punya ConvNeXtV2.

### 2. Cell `PREPROCESSING & AUGMENTATION`
Normalize mean/std **tetap sama** (ImageNet standar `0.485/0.456/0.406`, `0.229/0.224/0.225`) — pretrained `.ms_in22k_ft_in1k` juga pakai statistik ImageNet. Tidak perlu diubah dari versi ConvNeXt.

### 3. Cell `BUILD MODEL` — ini perubahan paling penting
Rename class jadi `SwinWasteClassifier`, tambah `permute` sebelum `custom_conv2d`:

```python
class SwinWasteClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            EXPERIMENT_CONFIG["model"]["backbone"],
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feat_channels = self.backbone.num_features  # 768

        self.custom_conv2d = nn.Conv2d(feat_channels, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.custom_maxpool = nn.MaxPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self._init_head_weights()

    def _init_head_weights(self) -> None:
        # sama persis kayak ConvNeXt, gak berubah
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)          # (B, H, W, C) — channel-last!
        x = x.permute(0, 3, 1, 2)     # -> (B, C, H, W)  <-- WAJIB DITAMBAH
        x = self.relu(self.custom_conv2d(x))
        x = self.custom_maxpool(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = SwinWasteClassifier(NUM_CLASSES).to(device)
```

`custom_maxpool` (kernel_size=2) di output 7x7 bakal jadi 3x3 — masih valid, gak perlu diubah, tinggal `AdaptiveAvgPool2d(1)` yang beresin ukuran akhir.

### 4. Cell `TRAINING LOOP` (setup run)
```python
MODEL_NAME = "04_swin"
RUN_NAME = f"04_swin_{RUN_ID}"
```
Ganti prefix `03_convnext` → `04_swin` di 2 tempat ini. Sisanya (phase1/phase2 training loop) sama persis, karena logic freeze/unfreeze udah generic pakai `model.backbone.named_parameters()`.

### 5. Cell `EVALUATION` (learning curve plot)
Ganti judul plot doang:
```python
plt.suptitle("Learning Curve (Swin Tiny Waste Classifier - PyTorch)", ...)
```

### 6. Cell `SAVE CONFIG`
```python
EXPERIMENT_CONFIG["model_name"] = "swin_waste_classifier_pytorch"
EXPERIMENT_CONFIG["base_architecture"] = (
    "timm swin_tiny_patch4_window7_224.ms_in22k_ft_in1k (in22k+in1k pretrained, num_classes=0 feature extractor)"
)
```

### Bagian yang TIDAK perlu diubah
- Import cell, seed/device, data loader, train-val split — full sama.
- Augmentation pipeline (albumentations) — full sama.
- Class weight computation — full sama.
- Phase 1 & Phase 2 training loop logic (`run_phase`, freeze/unfreeze pattern) — generic, gak nyentuh nama layer spesifik.
- Inference & submission cell — full sama, cuma jalan pakai `model` yang udah beda arsitektur di baliknya.

## Ringkasan checklist
- [ ] `EXPERIMENT_CONFIG["model"]["backbone"]` → `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`
- [ ] `EXPERIMENT_CONFIG["model"]["fine_tune_at"]` → `140`
- [ ] Rename class `ConvNeXtWasteClassifier` → `SwinWasteClassifier`
- [ ] Tambah `x = x.permute(0, 3, 1, 2)` di `forward()` sebelum conv (paling gampang kelewat, bikin error shape)
- [ ] `MODEL_NAME` / `RUN_NAME` → `04_swin`
- [ ] Judul plot learning curve
- [ ] `model_name` & `base_architecture` di save config
