# Experiments and evaluation plan

## E0: quality and alignment
- [ ] Curate >=30 same-design image pairs (vary orientation and lighting).
- [ ] Compare no alignment vs ORB+RANSAC.
- [ ] Measure valid homographies, match/inlier counts, overlap, failure cases.
- [ ] Record 10 challenging failures with screenshots.

## E1: traditional anomaly baseline
- [ ] Fix image resolution and test blur / morphology ablations.
- [ ] Sweep threshold only on validation data.
- [ ] Evaluate image AUROC and false alarms per 100 normal samples.
- [ ] Report segmentation metrics on images with proper masks.

## E2: classification
- [ ] HOG+linear SVM; then pretrained MobileNet/ResNet feature probe.
- [ ] Split by PCB subset and maintain a fully untouched test set.
- [ ] Record per-class errors and latency.

## E3: one-class anomaly localization
- [ ] Train PaDiM or PatchCore from normal-only data.
- [ ] Study 5, 20, 100 normal-image regimes and multiple seeds.
- [ ] Compare per-image detection and pixel localization.

## E4: robustness and product
- [ ] Brightness/exposure shifts, blur, rotated/translated inputs.
- [ ] Test review times vs manual baseline.
- [ ] Measure CPU p50/p95 and (if available) cloud GPU performance.

| Experiment | Dataset/version | Method | Image AUROC | Pixel AUROC | p95 ms | Notes |
|---|---|---|---|---|---|---|
| placeholder | not yet run | ORB+diff | TBD | TBD | TBD | Never fabricate metrics |
