# Setup
1. clone this repo and cd into the directory for this repo
1. setup conda
   ```
   conda create -c conda-forge --prefix ./.conda rdkit
   conda activate ./.conda
   python -m pip install -r requirements.txt
   ```
# Training
1. To start afresh, run
   ```
   python main_qm9.py --n_epochs 1300 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999
   ```
   This creates a model in `outputs/edm_qm9`, but you can change this by setting `--exp_name "new_name"` in which case the model will output to `outputs/new_name`
1. to resume a run, suppose we have a folder `outputs/edm_qm9` which contains `args.pickle`, `generative_model_ema.npy`, `generative_model.npy` and `optim.npy`. Replace `<N>` with the epoch number after that model was checkpointed and run the following
   ```
   python main_qm9.py --resume outputs/edm_qm9 --start_epoch <N>
   ```

# Running evaluations
Suppose we have pretrained the model and assume we have a folder `path/to/pretrained_weights` which contains `args.pickle`, `generative_model_ema.npy`, `generative_model.npy` and `optim.npy`.

To calculate metrics, run 
```python eval_analyze.py --model_path path/to/pretrained_weights --n_samples 100 # you can change this to 10_000``` 

To create visualisations, run
```python eval_sample.py --model_path path/to/pretrained_weights --n_samples 100 # you can change this to 10_000```

# Changes I made
1. edited the XYZ-TO-SPHERE conversion script to use multithreading (so it is faster)
1. added qm7b dataset for unconditional generation --- to use this, run the command above with `--dataset qm7b`
1. fix model resumption code

# General comments
By default, we train the model which *explicitly adds hydrogens*; if we add the option `--remove_h` we train the model without hydrogens

# TODOs
Although the qm7b dataset has been added and allows the model to be correctly trained, it does not yet allow the model to be correctly evaluated. The evaluation code currently still assumes we are using the qm9 dataset. Need to update this! Mainly `qm9/analyze.py` and `qm9/visualiser.py`.