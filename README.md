## Active Learning in Bayesian Neural Networks with Balanced Entropy Learning Principle (ICLR 2023)

This repository contains PyTorch code for Balanced Entropy Learning. Most of the code in this repository has been adapted from [here](https://github.com/BlackHC/BatchBALD).

For details, see [Active Learning in Bayesian Neural Networks with Balanced Entropy Learning Principle ](https://openreview.net/forum?id=ZTMuZ68B1g) by Jae Oh Woo.

#### Setup
1. Install torch
2. Install requirements

#### Example
```console
❱❱❱ python ./src/run_experiment.py --experiment_task_id mnist_independent_balentacq --experiment_description mnist_independent_balentacq --dataset mnist --initial_sample 25 --seed 987654321 --num_inference_samples 100 --available_sample_k 25 --type balentacq --acquisition_method independent --batch_size 16 --epochs 150 --target_accuracy 0.9999 --target_num_acquired_samples 300 --scoring_batch_size 1024 --test_batch_size 100 --validation_set_size 100 --gpu-device 0
```

#### Citation
If you find that Balanced Entropy interesting and help your research, please consider citing it:
```
@inproceedings{
woo2023active,
title={Active Learning in Bayesian Neural Networks with Balanced Entropy Learning Principle},
author={Jae Oh Woo},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=ZTMuZ68B1g}
}
```
