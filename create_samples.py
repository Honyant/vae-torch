import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


from net import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_latent = 512

vae = VAE(n_latent)
vae.apply(init_weights)
vae = vae.to(device)
vae.load_state_dict(torch.load('vae.pth'))

dataset1 = datasets.CelebA('../data', split='train', download=False,
                    transform=transforms.Compose([transforms.ToTensor()]))
# Interpolate between two points in latent space
def interpolate(start, end, steps):
    interpolation = np.zeros((steps, n_latent))
    for i in range(steps):
        interpolation[i] = start * (1 - i / steps) + end * i / steps
    return interpolation

# Generating two random points in the latent space
z_start = torch.randn(1, n_latent)
z_end = torch.randn(1, n_latent)

steps = 10  # For a 10x10 grid
interpolations = interpolate(z_start, z_end, steps)

# Generating a batch of all interpolated latent vectors
all_interpolated = []
for row in range(steps):
    z_row_start = interpolations[row]
    z_row_end = interpolations[(row + 1) % steps]  # Loop around for the last row
    z_row_interpolated = interpolate(z_row_start, z_row_end, steps)
    all_interpolated.extend(z_row_interpolated)

img_shape = dataset1[0][0].shape
# Process the entire batch
all_interpolated = np.array(all_interpolated)
batch_z = torch.tensor(all_interpolated).float().to(device)
print(batch_z.shape)
batch_z= torch.randn(*batch_z.shape).to(device)*1.5
batch_recon = vae.decode(batch_z).permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()

# Arrange the images into one large grid image
# grid_image = np.zeros((28 * steps, 28 * steps))
grid_image = np.zeros((img_shape[1] * steps, img_shape[2] * steps, img_shape[0]))

for i, img in enumerate(batch_recon):
    row = i // steps
    col = i % steps
    # grid_image[row*28:(row+1)*28, col*28:(col+1)*28] = img
    grid_image[row*img_shape[1]:(row+1)*img_shape[1], col*img_shape[2]:(col+1)*img_shape[2], :] = img

# Display the large grid image
plt.figure(figsize = (40, 40))
plt.imshow(grid_image, cmap='gray')
plt.axis('off')
plt.savefig('interpolations.png', bbox_inches='tight', pad_inches=0)
plt.close() # Close the figure to free memory
# Save the large grid image