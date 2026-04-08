# 🔍 Crack Segmentation with U-Net (PyTorch)

Binary semantic segmentation of surface cracks using a custom 4-level U-Net architecture, trained from scratch with PyTorch.

---

## 📊 Results

| Metric    | Score  |
|-----------|--------|
| IoU       | 0.5293 |
| Dice      | 0.6649 |
| Precision | 0.6040 |
| Recall    | 0.8141 |
| F1        | 0.6649 |

> Best inference threshold: **0.90** (determined via threshold sweep on validation set)

---

## 🏗️ Architecture

- 4-level U-Net encoder-decoder
- Filter progression: 64 → 128 → 256 → 512 → 1024 (bottleneck)
- Learnable upsampling via `ConvTranspose2d`
- Dropout (0.3) in bottleneck
- ~31M trainable parameters

## ⚙️ Training Details

- **Loss:** BCE with logits (pos_weight=10.0) + Dice Loss (30/70 split)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing (T_max=60, eta_min=1e-7)
- **Batch size:** 8
- **Image size:** 256×256
- **Mixed precision:** `torch.amp.autocast("cuda")`
- **Early stopping:** patience=10
- **Best epoch:** 37

## 📦 Dataset

[Crack Segmentation Dataset — Kaggle](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset)

Expected structure after download:

crack_segmentation_dataset/
├── train/
│   ├── images/
│   └── masks/
└── test/
├── images/
└── masks/

## 🚀 Setup

```bash
pip install torch torchvision albumentations opencv-python-headless tqdm segmentation-models-pytorch
```

## 📁 Project Structure

```
crack_segmentation/
├── crack_segmentation.ipynb        # Main notebook
├── crack-segmentation-dataset.zip  # Dataset (not pushed to GitHub)
├── crack_segmentation_dataset/     # Extracted automatically by notebook
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── README.md
└── .gitignore
```

> ⚠️ Model weights (`best_crack_model_pytorch.pth`) and the dataset zip are not included in this repository due to file size. Download the dataset from Kaggle and train using the notebook.

---

## 🔒 Security Notes

- No API keys, credentials, or personal tokens are stored in this repository
- Dataset paths in the notebook use local Windows paths — update `ZIP_PATH` and `EXTRACT_DIR` in Cell 5 to match your own system before running

---

## 📄 License

This project is for educational and research purposes.