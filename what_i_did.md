1. clone this repo
   ```
   git clone https://github.com/ehoogeboom/e3_diffusion_for_molecules
   cd e3_diffusion_for_molecules
   ```
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
1. customize it
   ```
   python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 100 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 2e-3 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --dataset qm9_second_half
   ```

   ```
   python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 100 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 512 --nf 128 --n_layers 3 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --dataset qm9_second_half
   ```

# Processing XYZ
Charges: atom to proton number (tensor of size num_atoms)
Positions: tensor of size num_atoms x 3 (x y z)