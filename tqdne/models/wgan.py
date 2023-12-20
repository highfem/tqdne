import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WGenerator(nn.Module):
    def __init__(self, wave_size, latent_dim, channels=1, dim=16):
        super(WGenerator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.wave_size = wave_size
        #self.feature_sizes = (self.wave_size[0] / 16, self.wave_size[1] / 16)
        self.feature_size = int(wave_size / 16)
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_size),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose1d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm1d(4 * dim),
            nn.ConvTranspose1d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * dim),
            nn.ConvTranspose1d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.ConvTranspose1d(dim, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_size)
        # Return generated image
        x = self.features_to_image(x)
        return x

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class WDiscriminator(nn.Module):
    def __init__(self, wave_size, channels, dim=16):
        """
        wave_size : int E.g. 1024
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(WDiscriminator, self).__init__()

        self.wave_size = wave_size

        self.image_to_features = nn.Sequential(
            nn.Conv1d(channels, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * int(wave_size / 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)