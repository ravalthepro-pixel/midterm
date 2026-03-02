# Results

This folder contains all output files generated automatically by running `train.py`.

| File | Description |
|---|---|
| `best_model.pth` | Best model weights checkpoint (highest validation accuracy) |
| `training_curves.png` | Loss & accuracy curves across all 20 training epochs |
| `confusion_matrix.png` | 10×10 heatmap of true vs. predicted digit classes |
| `sample_predictions.png` | Grid of 10 validation images with true/predicted labels (green=correct, red=wrong) |
| `classification_report.txt` | Per-class precision, recall, F1-score, and support |

## How to Generate

```bash
python train.py
```

All files will be saved to this `results/` directory automatically.

## Authors
Jasraj "Jay" Raval & Jack Sweeney — AI 100 Midterm Project
