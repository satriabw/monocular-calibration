# Hasselt automatic calibration: GPU benchmark (2026-10-30)

## Result

**Rejected (exit 2).** Full RoMa on the RTX 2080 Ti produced only **2 matches**
(confidence 0.21-0.24); the pipeline requires at least 20 finite, distinct ground
correspondences. The failure is in matching, not in the environment. Thresholds
were not loosened.

Command:

```bash
python auto_calibrate.py --image data/hasselt/Background.bmp \
  --location 50.926946 5.336048 --reference-width 40 \
  --matcher roma --output-dir results/hasselt-gpu
```

Notes from the run:
- AnyCalib intrinsics: f = 1231.9 px, principal point (639.5, 360.4), no distortion.
- Reference: 40 m Digitaal Vlaanderen orthophoto, 1000x1000 px (4 cm/px).
- Output: `results/hasselt-gpu/` (`quality_report.json`, `matches_0.csv`,
  `intrinsics_0.json`, `reference.png`). No `camera_model.yml` (failure).
- Tests: 14 passed.

## Environment (conda)

```bash
conda create -n monocular-calibration python=3.12
export PIP_CONFIG_FILE=/dev/null   # see issue 1
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
pip install -r requirements-auto.txt
```

Install torch (cu121) **first**; `requirements-auto.txt` alone pulls a CUDA 13 build.

## Issues hit

1. **pip stalls/fails.** `~/.config/pip/pip.conf` (NVIDIA PyIndex) adds
   `pypi.ngc.nvidia.com` as an extra index; it does not resolve on this machine.
   Workaround: `PIP_CONFIG_FILE=/dev/null`. Config left unchanged.
2. **torch built for CUDA 13.** Driver 535 supports CUDA 12.2, so
   `torch.cuda.is_available()` was False. Fixed with the cu121 build.
3. **`No module named 'local_corr'`.** RoMa's optional fused CUDA kernel
   (`fused-local-corr`) is imported on GPU. The prebuilt wheel is ABI-incompatible
   with torch 2.5.1, and it cannot be built (system `nvcc` is 10.1). Installing it
   also replaced torch with a CUDA 13 build and broke cuDNN
   (`CUDNN_STATUS_NOT_INITIALIZED`), so the env was rebuilt. RoMa itself does not
   need CUDA 13.

## Code change

`auto_calibrate.py` (`roma_model`): use the fused kernel only if `local_corr`
imports; otherwise fall back to RoMa's native-torch path (same result, likely
slower). Uncommitted.

## Open question

Why only 2 matches? Candidates to check: reference/camera viewpoint gap, the
winter orthophoto vs. scene content, the 40 m crop width, RoMa confidence/sampling
settings, and whether the native-torch correlation path behaves identically to the
fused kernel. Not investigated yet.
