# Full Waveform Inversion — Seismic-to-Velocity CNN
 
An encoder-decoder CNN for Kaggle's **Waveform Inversion** competition: predict 2D subsurface velocity maps directly from 4D seismic waveform recordings, across three different synthetic data families (Vel, Fault, Style).
 
![Architecture](images/architecture_flow.png)
 
---
 
## Table of Contents
 
- [The task](#the-task)
- [Why this architecture](#why-this-architecture)
- [Design decisions](#design-decisions)
- [Repo structure](#repo-structure)
- [Running it](#running-it)
- [Validation status](#validation-status)
---
 
## The task
 
Full Waveform Inversion (FWI) reconstructs subsurface structure from seismic wave recordings. Traditional physics-based solvers are accurate but slow and noise-sensitive; pure ML is fast but data-hungry and generalizes poorly. This model takes the supervised-ML side of that trade-off: given a waveform tensor `(sources, time_steps, receivers)`, predict the corresponding velocity map `(height, width)` directly, end to end, no physics solver in the loop.
 
Evaluation is MAE, computed only on odd-indexed `x_` columns but every `y_` row, stacked across all test samples (`oid`s).
 
## Why this architecture
 
The natural instinct is "use a U-Net" — but U-Net's skip connections assume the input and output are pixel-aligned in the same spatial domain. Here they aren't: the input is a *time-series-over-receivers* waveform tensor, the output is a *spatial* velocity map. An early-layer waveform feature has no meaningful spatially-aligned counterpart in the velocity map, so skip connections would just wire together unrelated representations.
 
Instead, this follows the **InversionNet** approach (the standard baseline architecture for OpenFWI-derived tasks, which this competition's data is built from): a strided-convolution encoder collapses the waveform tensor into a compact latent representation, and a transposed-convolution decoder expands that latent representation into the velocity map from scratch — no cross-connections back to the encoder.
 
## Design decisions
 
- **L1 loss, not L2** — matches the competition's MAE metric directly, and doesn't over-smooth the sharp velocity discontinuities at geological boundaries the way L2 tends to.
- **Shape-agnostic by construction.** Fault-family geometry isn't guaranteed to match Vel/Style. Rather than hardcoding tensor sizes, the encoder ends in an `AdaptiveAvgPool2d` bottleneck and the decoder ends in `F.interpolate` to the exact target `(H, W)` — the same model handles whatever `(S, T, R)` → `(H, W)` shapes it's given, inferred from the data at training time.
- **File-level, family-stratified train/val split.** Splitting at the sample level would leak near-duplicate structure between train and val, since many samples in the same source file share generation-process quirks. Splitting whole files, stratified by family, avoids that while still keeping each family represented in both splits.
- **Lazy memory-mapped loading.** Each real training file holds ~500 samples and can run several hundred MB; the dataset indexes every sample across every file but only memory-maps files (`np.load(..., mmap_mode="r")`) rather than loading them eagerly, so this doesn't blow out RAM.
- **Mixed precision + AdamW + cosine schedule** — standard, appropriate for a single T4/P100-class GPU budget on Kaggle's free tier.
- **Submission columns read from `sample_submission.csv` itself**, not assumed — `predict.py` doesn't hardcode "odd columns of a 70-wide map"; it parses the actual expected `x_` column names and order out of the real file, so it stays correct even if the true format differs subtly from the competition description's example.
## Repo structure
 
```
.
├── README.md
├── requirements.txt
├── images/
│   └── architecture_flow.png
└── src/
    ├── data.py       # file discovery, lazy-loading Dataset, family-stratified split
    ├── model.py       # InversionNet-style encoder-decoder
    ├── train.py       # training loop, checkpointing on best val MAE
    └── predict.py      # inference + submission CSV generation
```
 
## Running it
 
On a Kaggle notebook (GPU on):
 
```python
!python src/train.py \
  --root /kaggle/input/waveform-inversion/train_samples \
  --epochs 20 --batch-size 16 --out checkpoints/
 
!python src/predict.py \
  --checkpoint checkpoints/best.pt \
  --test-dir /kaggle/input/waveform-inversion/test \
  --sample-submission /kaggle/input/waveform-inversion/sample_submission.csv \
  --out submission.csv
```
 
Locally:
 
```bash
git clone https://github.com/<your-username>/fwi-velocity-cnn.git
cd fwi-velocity-cnn
pip install -r requirements.txt
```
 
## Validation status
 
I don't have access to the real competition data, so this has been validated for **mechanical correctness**, not modeling performance: a synthetic dataset was built matching the real shapes and file-naming conventions for all three families (Vel, Style, Fault) plus un-batched test files and a real-format `sample_submission.csv`. Against that synthetic data, confirmed end to end:
 
- File discovery correctly finds and pairs all three families' naming conventions
- Dataset loading, normalization, and shape handling work correctly
- Model forward pass produces the correct output shape and respects the `[-1, 1]` tanh range
- The full training loop runs, trains, and checkpoints without error
- Inference produces a submission file that **exactly matches** `sample_submission.csv`'s columns and row-for-row `oid_ypos` values, with predicted values in a physically sensible velocity range
What's *not* yet known: real-world MAE, whether 20 epochs/these hyperparameters are well-tuned, and whether the Fault family needs different handling than Vel/Style once trained on real signal rather than noise.
 
---
 
*Built for Kaggle's Waveform Inversion competition. Data derived from [OpenFWI](https://smileunc.github.io/projects/openfwi/datasets).*
 
