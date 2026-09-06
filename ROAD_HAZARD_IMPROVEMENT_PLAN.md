# Road-hazard improvement experiment

## Baseline to retain

Keep Member 2's final YOLO11n model and its untouched validation/test metrics. Do not overwrite the baseline checkpoint or alter the baseline test set.

## Variant hypothesis

Pothole and crack performance may improve when their training images are oversampled and the model sees controlled road-appearance variation. Manhole images are not duplicated because that class is already substantially stronger.

## Create the variant

After any active Python/GPU job finishes, run from `C:\Users\user\Downloads\Dataset`:

```powershell
python SafeDrive\tools\create_hard_hazard_variant.py `
  --source SafeDrive_Dataset `
  --output SafeDrive_Dataset_hardclass
```

This adds one duplicate only for training images that contain pothole and/or crack. Validation and test images/labels are unchanged.

## Train the variant

```powershell
python SafeDrive\tools\train_road_hazard_variant.py `
  --data SafeDrive_Dataset_hardclass\data.yaml `
  --project SafeDrive_Results `
  --name yolo11n_hardclass_aug
```

The batch-size default is 4 for a laptop RTX 4050 at 960 px. Lower it to 2 if CUDA runs out of memory.

## Decision rule

Evaluate both models on the same untouched test set. Adopt the variant only if pothole/crack mAP50 or recall improves without an unacceptable rise in normal-image false positives. Keep the baseline otherwise and report the negative result as an ablation.
