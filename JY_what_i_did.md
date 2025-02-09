1. clone this repo and cd into the directory for this repo
1. setup conda
   ```
   conda create -c conda-forge --prefix ./.conda rdkit
   conda activate ./.conda
   python -m pip install -r requirements.txt
   ```
1. run
   ```
   python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999
   ```
1. to resume a run
   ```
   python main_qm9.py --resume outputs/edm_qm9 --start_epoch 338
   ```


# Changes I made
1. edited the XYZ-TO-SPHERE conversion script to use multithreading (so it is faster)
1. added qm7b dataset for unconditional generation
1. fix model resumption code
