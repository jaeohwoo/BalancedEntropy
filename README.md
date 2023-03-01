## Active Learning in Bayesian Neural Networks with Balanced Entropy Learning Principle (ICLR 2023)

#### Example
```console
❱❱❱ python ./src/run_experiment.py --experiment_task_id mnist_independent_balentacq --experiment_description mnist_independent_balentacq --dataset mnist --initial_sample 25 --seed 987654321 --num_inference_samples 100 --available_sample_k 25 --type balentacq --acquisition_method independent --batch_size 16 --epochs 150 --target_accuracy 0.9999 --target_num_acquired_samples 300 --scoring_batch_size 1024 --test_batch_size 100 --validation_set_size 100 --gpu-device 0
```


#### Code base: https://github.com/BlackHC/BatchBALD
