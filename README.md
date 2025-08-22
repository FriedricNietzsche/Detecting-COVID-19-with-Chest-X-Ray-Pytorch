# Detecting COVID-19 with Chest X-Ray (PyTorch)

Image classification of chest X-rays into three classes: Normal, Viral Pneumonia, and COVID-19, using transfer learning (ResNet-18) in PyTorch.

This repo contains a single Jupyter Notebook that prepares the dataset, defines a custom Dataset/DataLoader, fine-tunes a model, and visualizes predictions.

## Project structure

```
.
├─ Covid-19 Detection With Chest X-Ray.ipynb
└─ COVID-19 Radiography Database/
   ├─ Normal/ (or Normal/images)
   ├─ Viral Pneumonia/ (or Viral Pneumonia/images)
   └─ COVID/ (or COVID/images)
```

Notes:
- The dataset in Kaggle may have images either directly under the class folder or under a nested `images/` folder. The notebook handles both.
- The notebook creates a small test split at `COVID-19 Radiography Database/test/{normal,viral,covid}` on first run by moving up to 30 images per class.

## Requirements
- Python 3.10+
- PyTorch and TorchVision (versions shown are examples; the notebook installs/uses those available in your environment)
- NumPy, Matplotlib, Pillow

Example (Windows PowerShell):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib pillow tqdm scikit-learn
```

If you do not have a CUDA-capable GPU, install the CPU wheel instead (omit the `--index-url`).

## Dataset
- Source: COVID-19 Radiography Dataset (Kaggle)
- Place the unzipped dataset folder at `COVID-19 Radiography Database/` in the project root (matching the notebook’s paths).
- The notebook detects and supports both of these layouts per class:
  - `COVID-19 Radiography Database/Normal/*.png` (or jpg/jpeg)
  - `COVID-19 Radiography Database/Normal/images/*.png`

On first run, the notebook will create a small test split by moving up to 30 images per class to:
```
COVID-19 Radiography Database/test/normal
COVID-19 Radiography Database/test/viral
COVID-19 Radiography Database/test/covid
```

## What the notebook does
1. Imports libraries and sets seeds for reproducibility.
2. Prepares a small test split (idempotent: skipped if already exists).
3. Defines a robust `ChestXRayDataset` that supports nested `images/` folders and multiple extensions.
4. Defines train/test transforms (ImageNet normalization).
5. Builds DataLoaders.
6. Initializes ResNet-18 with ImageNet weights, replaces the final layer for 3 classes.
7. Trains briefly and evaluates, visualizing predictions.

## How to run
1. Open `Covid-19 Detection With Chest X-Ray.ipynb` in VS Code or Jupyter.
2. Ensure the dataset folder is present as described above.
3. Run cells top to bottom.

Tips:
- If you see a warning about `pretrained=True`, prefer the newer API: `from torchvision.models import ResNet18_Weights; torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- If you have a GPU, set device in the notebook and move model/batches to CUDA for speed.

## Improvements (roadmap)
- Add a proper validation split (avoid evaluating on test during training).
- Deterministic dataset indexing (remove random class choice in `__getitem__`).
- Device handling (CUDA/CPU), mixed precision (AMP), and better loaders (`num_workers`, `pin_memory`).
- Learning rate scheduler and early stopping.
- Class imbalance handling (weighted sampler or class weights).
- Metrics: confusion matrix, per-class precision/recall/F1, and model checkpointing.
- Extract training/eval into small Python modules for reuse; keep the notebook as a runner.

## Troubleshooting
- File not found for test dirs: run the dataset-prep cell again and ensure your dataset path matches `COVID-19 Radiography Database/`.
- If train batches are zero: verify class folders exist and contain images (png/jpg/jpeg). Check nested `images/` subfolders.
- TorchVision weights download issues: ensure internet access or pre-download weights/cache.

## License
This project is for educational purposes. Respect the Kaggle dataset license/terms when using the data.
