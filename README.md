# Active Learning in Bayesian Neural Networks with **Balanced Entropy** (ICLR 2023)

[![Paper (OpenReview)](https://img.shields.io/badge/Paper-OpenReview-black)](https://openreview.net/forum?id=ZTMuZ68B1g)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![ICLR 2023](https://img.shields.io/badge/Venue-ICLR%202023-blue)](https://openreview.net/forum?id=ZTMuZ68B1g)

This repository provides a clean PyTorch reference implementation of **Balanced Entropy** for active learning with Bayesian neural networks (BNNs).
Much of the codebase was adapted from the excellent [BatchBALD](https://github.com/BlackHC/BatchBALD) project.

> A practical, reproducible implementation of **Balanced Entropy** acquisition for uncertainty-aware sample selection in BNNs.

---

## Update — August 2025

* **Fixed:** `marginalized_posterior_entropy` in `torch_utils.py`.
* **Important:** This bug was **introduced during a later code cleanup** and **does not affect any experiments or results reported in the ICLR 2023 paper**.
  The paper’s results were produced with pipelines/snapshots unaffected by this issue.

---

## Setup

1. **Install PyTorch** (use the command appropriate for your OS/CUDA):

   * See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Example

```bash
python ./src/run_experiment.py \
  --experiment_task_id mnist_independent_balentacq \
  --experiment_description mnist_independent_balentacq \
  --dataset mnist \
  --initial_sample 25 \
  --seed 987654321 \
  --num_inference_samples 100 \
  --available_sample_k 25 \
  --type balentacq \
  --acquisition_method independent \
  --batch_size 16 \
  --epochs 150 \
  --target_accuracy 0.9999 \
  --target_num_acquired_samples 300 \
  --scoring_batch_size 1024 \
  --test_batch_size 100 \
  --validation_set_size 100 \
  --gpu-device 0
```

---

## Repository Structure (high level)

```
src/
  ├─ acquisition/           # Balanced Entropy and related acquisition functions
  ├─ datasets/              # Dataset wrappers and utilities
  ├─ models/                # BNN models and inference utilities
  ├─ utils/                 # Common utilities and helpers
  ├─ torch_utils.py         # (fixed) entropy/uncertainty routines
  └─ run_experiment.py      # End-to-end experiment driver
```

---

## FAQ

**Does the August 2025 fix change the paper’s results?**
No. The bug was introduced **after** the paper as part of a code cleanup and **did not exist** in the pipelines used for the ICLR 2023 experiments.

**What’s the relationship to BatchBALD?**
Data handling and several utilities are adapted from BatchBALD; acquisition logic and utilities are reorganized/extended for Balanced Entropy.

**Can I use this on datasets beyond MNIST?**
Yes. The components are modular; you can plug in other image datasets (e.g., CIFAR) and compatible BNN backends with minimal changes.

---

## Citation

If you find Balanced Entropy useful, please cite:

```bibtex
@inproceedings{
woo2023active,
title={Active Learning in Bayesian Neural Networks with Balanced Entropy Learning Principle},
author={Jae Oh Woo},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=ZTMuZ68B1g}
}
```

---

## Acknowledgments

* Portions of the codebase are adapted from [BatchBALD](https://github.com/BlackHC/BatchBALD).

---

## License

See `LICENSE` for details.

---
