import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from net import *
# make the images subdirectory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")
# Hyperparameters
batch_size = 32
test_batch_size = 32
n_latent = 512
learning_rate = 5e-3
num_epochs = 20
save_model = True

transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset1 = datasets.CelebA('../data', split='train', download=False,
                    transform=transform)
dataset2 = datasets.CelebA('../data', split='test', download=False,
                    transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size, shuffle=True)


# Hyperparameters
batch_size = 32
test_batch_size = 32
n_latent = 512
learning_rate = 5e-4
num_epochs = 10
save_model = True


vae = VAE(n_latent)
vae.apply(init_weights)
vae = vae.to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))


def dice(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


kl_divergence_weight = n_latent * 0.1 / (dataset1[0][0].shape[0] * dataset1[0][0].shape[1] * dataset1[0][0].shape[2])
image_save_dir = "images"
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = vae(data)

        # Loss calculation
        logvar_exp = logvar.exp().clamp(max=1e6, min=1e-6) # Clamping for numerical stability

        recon_loss = F.mse_loss(recon_batch, data, reduction='mean')
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar_exp, dim=1), dim=0)
        dice_loss = dice(recon_batch, data)
        loss = recon_loss + kl_divergence * kl_divergence_weight + dice_loss * 0.1

        # Backward pass and optimization
        loss.backward()

        torch.nn.utils.clip_grad_norm_(vae.parameters(), 5)

        optimizer.step()
        scheduler.step()

        # Periodic logging and image generation
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, '
                f'Recon: {recon_loss.item()}, KL: {kl_divergence.item()}, Dice: {dice_loss.item()}')
            
            with torch.no_grad():
                # Visualizing and saving reconstruction
                sample_recon = recon_batch[0].permute(1, 2, 0).cpu().numpy()
                sample_orig = data[0].permute(1, 2, 0).cpu().numpy()
                combined_recon = np.concatenate((sample_orig, sample_recon), axis=1)

                recon_image_file = os.path.join(image_save_dir, f'epoch_{epoch}_batch_{batch_idx}_recon.png')
                plt.imsave(recon_image_file, combined_recon, cmap='gray')

                # Generating, visualizing, and saving a random latent vector image
                random_z = torch.randn(1, n_latent).to(device)
                random_recon = vae.decode(random_z).detach().permute(0, 2, 3, 1).squeeze().cpu().numpy()

                random_image_file = os.path.join(image_save_dir, f'epoch_{epoch}_batch_{batch_idx}_random.png')
                plt.imsave(random_image_file, random_recon, cmap='gray')

if save_model:
    timestamp = time.strftime("%y_%m_%d_%H_%M_%S")
    torch.save(vae.state_dict(), f"vae_cnn_{num_epochs}_{timestamp}.pth")