# Environment notes

The following are inventories of the local environments as observed on 2026-11-14, not verified historical training environments or portable lockfiles.

- Detector: Windows 64-bit, Python 3.10.20, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, NumPy 2.2.6. Full installed-package inventory: `detector-installed-packages.txt`.
- Tooth-prior preprocessing: Windows 64-bit, Python 3.10.20, PyTorch 2.7.0+cu128, Ultralytics 8.4.116, OpenCV 5.0.0, NumPy 2.2.6. Full installed-package inventory: `tooth-prior-installed-packages.txt`.

`*-conda-packages.txt` additionally records package names, versions and build strings from each local conda environment; it does not record channels.

The inventories include unrelated research packages and may include local or platform-specific installations; they should not be installed wholesale. Exact historical training dependencies and clean installation remain to be verified. CUDA runtime compatibility depends on the local driver and operating system.

Suggested starting point in a clean Python 3.10 environment:

```sh
python -m pip install -r requirements.txt  # detector and offline tooth-prior generation
```

A candidate tooth segmenter checkpoint and train/evaluation entry point are included. Its provenance relative to the final detector experiment and the applicable license terms require verification. Datasets, generated priors and detector weights are not included. Do not claim that these observed package versions reproduce the paper's metrics without testing the final experiment.

## Same-environment check

On 2026-11-14 the local tooth-prior environment successfully imported and constructed both GraphRTDETR and a YOLO segmentation model using the local segmenter checkpoint. Its PyTorch, torchvision, Ultralytics and OpenCV versions match the main requirements. TensorBoard is not installed in that existing environment; it is included in the combined installation requirements. No end-to-end training or clean-environment installation has been validated.
