import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdne.utils.positionalencoding import positional_encoding


class WGenerator(nn.Module):
    def __init__(self, wave_size, latent_dim, encoding_L=4, num_vars=1, out_channels=1, dim=16):
        super(WGenerator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.wave_size = wave_size

        self.label_embeddings = lambda x: torch.flatten(
            positional_encoding(x, encoding_L), start_dim=1
        )
        latent_dim += 2 * encoding_L * num_vars
        # self.feature_sizes = (self.wave_size[0] / 16, self.wave_size[1] / 16)
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
            nn.ConvTranspose1d(dim, out_channels, 4, 2, 1),
        )

    def forward(self, input_data, cond=None):
        # Map latent into appropriate size for transposed convolutions
        x = input_data
        if cond is not None:
            cond = self.label_embeddings(cond)
            x = torch.cat([x, cond], dim=1)

        x = self.latent_to_features(x)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_size)
        # Return generated image
        x = self.features_to_image(x)
        return x

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class WDiscriminator(nn.Module):
    def __init__(self, wave_size, in_channels, encoding_L=4, num_vars=1, dim=16):
        """
        wave_size : int E.g. 1024
        """
        super(WDiscriminator, self).__init__()

        self.wave_size = wave_size

        self.label_embeddings = lambda x: torch.flatten(
            positional_encoding(x, encoding_L), start_dim=1
        )
        self.in_channels = in_channels # + 2 * encoding_L * num_vars

        self.image_to_features = nn.Sequential(
            nn.Conv1d(self.in_channels, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4 * dim, 8 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2)
            # nn.Sigmoid()
        )
        
        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * int(wave_size / 16) + 2 * encoding_L * num_vars
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, input_data, cond=None):
        batch_size = input_data.size()[0]
        x = input_data.view(batch_size, self.in_channels, -1)
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        if cond is not None:
            cond = self.label_embeddings(cond)
            x = torch.cat([x, cond], dim=1)        
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
