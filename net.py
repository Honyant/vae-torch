import torch
import torch.nn as nn
import torch.nn.functional as F

def lrelu(x, alpha=0.3):
    return F.leaky_relu(x, negative_slope=alpha)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class VAE(nn.Module):
    def __init__(self, n_latent = 512):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1)
        self.resEncode = nn.ModuleList([ResidualBlock(256, 256) for _ in range(3)])
        self.resDecode = nn.ModuleList([ResidualBlock(256, 256) for _ in range(3)])

        self.fc1 = nn.Linear(256 * 28 * 23, n_latent)
        self.fc2 = nn.Linear(256 * 28 * 23, n_latent)
        # init fc2 weights to be 0 using weight initialization
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

        # Decoder
        self.dec_fc = nn.Linear(n_latent, 256 * 28 * 23)
        nn.init.constant_(self.dec_fc.weight, 0.0001)
        nn.init.constant_(self.dec_fc.bias, 0.0001)
        self.dec_conv2d_t1 = nn.ConvTranspose2d(256, 96, kernel_size=3, stride=2, padding=1)
        self.dec_conv2d_t2 = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1)
        self.dec_conv2d_t3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = lrelu(self.enc_conv1(x))
        x = F.dropout(x, 0.2)
        x = lrelu(self.enc_conv2(x))
        x = F.dropout(x, 0.2)
        x = lrelu(self.enc_conv3(x))
        x = F.dropout(x, 0.2)
        for layer in self.resEncode:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return lrelu(self.fc1(x)), self.fc2(x)

    def reparameterize(self, mu, logvar):
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = lrelu(self.dec_fc(z))
        x = x.view(x.size(0), 256, 28, 23)
        for layer in self.resDecode:
            x = layer(x)
        x = lrelu(self.dec_conv2d_t1(x))
        x = F.dropout(x, 0.2)
        x = lrelu(self.dec_conv2d_t2(x))
        x = F.dropout(x, 0.2)
        x = torch.sigmoid(self.dec_conv2d_t3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Model, Loss and Optimizer
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.02, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
