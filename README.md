# VAE Face Generation

This project trains a Variational Autoencoder (VAE) on the CelebA dataset to generate facial images.

## Setup

1. **Install Dependencies**: Ensure you have Python 3.x installed and then install the required Python packages:

```bash
pip install -r requirements.txt
```

2. **Data Preparation**: Download the CelebA dataset and place it in `../data`. The script expects this specific structure.

## Training

To start the VAE training, run:

```bash
./train.sh
```

This script uses `screen` to run training in a detached session. Use `screen -r vae_training_run` to reattach to the session.

## Sampling

After training, generate interpolated samples with:

```bash
python create_samples.py
```

This script loads the trained model, generates interpolations between two latent points, and saves the output as `interpolations.png`.

Sample generation:
![picture 0](https://i.imgur.com/shXHcT8.jpeg)  


## Utilities

- **tmux Sessions**: If you prefer `tmux` over `screen`, you can use the provided `tmux` commands within `train.sh` for session management.
- **Reattach to Training Session**: Use the provided `reattach_to_vae_training_run.sh` script tailored for `tmux`.

## Note

Ensure CUDA is available for GPU acceleration, falling back to CPU if not present.


tmux new-session -d -s vae_training_run "python main.py"
echo "tmux VAE training started."


tmux attach-session -t vae_training_run
