import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, log2
from tqdne.utils.positionalencoding import positional_encoding


class CGenerator(nn.Module):
    r"""Conditional GAN (CGAN) generator based on a DCGAN model from
    `"Conditional Generative Adversarial Nets
    by Mirza et. al. " <https://arxiv.org/abs/1411.1784>`_ paper

    Args:
        num_classes (int): Total classes present in the dataset.
        encoding_dims (int, optional): Dimension of the encoding vector sampled from the noise prior.
        out_size (int, optional): Height and width of the input image to be generated. Must be at
            least 16 and should be an exact power of 2.
        out_channels (int, optional): Number of channels in the output Tensor.
        step_channels (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
        batchnorm (bool, optional): If True, use batch normalization in the convolutional layers of
            the generator.
        nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the intermediate
            convolutional layers. Defaults to ``LeakyReLU(0.2)`` when None is passed.
        last_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the final
            convolutional layer. Defaults to ``Tanh()`` when None is passed.
    """

    def __init__(
        self,
        num_variables,
        latent_dim=128,
        encoding_L=4,
        out_size=1024,
        out_channels=2,
        step_channels=32,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="generated",
    ):
        super(CGenerator, self).__init__()
        self.encoding_input_dims = latent_dim
        self.num_variables = num_variables
        self.label_embeddings = lambda x: torch.flatten(
            positional_encoding(x, encoding_L), start_dim=1
        )
        self.encoding_input_cond_dims = latent_dim + 2 * encoding_L * num_variables
        self.label_type = label_type
        if out_size < 16 or ceil(log2(out_size)) != log2(out_size):
            raise Exception(
                "Target Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = out_size.bit_length() - 4
        self.ch = out_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        model = []
        d = int(self.n * (2**num_repeats))
        if batchnorm is True:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        self.encoding_input_cond_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nn.BatchNorm1d(d),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm1d(d // 2),
                        nl,
                    )
                )
                d = d // 2
        else:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        self.encoding_input_cond_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nl,
                    )
                )
                d = d // 2

        model.append(
            nn.Sequential(nn.ConvTranspose1d(d, self.ch, 4, 2, 1, bias=True), last_nl)
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def sampler(self, sample_size, cond_list, device):
        mean = torch.zeros(sample_size, self.encoding_input_dims)
        z = torch.normal(mean=mean).to(device)
        cond_list_tensor = torch.tensor(cond_list, dtype=torch.float32, device=device)
        return z, cond_list_tensor

    def forward(self, z, y):
        r"""Calculates the output tensor on passing the encoding ``x`` through the Generator.

        Args:
            x (torch.Tensor): A 1D torch tensor of the encoding sampled from a probability
                distribution.

        Returns:
            A 2D torch.Tensor of the generated signal.
        """
        z1 = z.unsqueeze(2)
        #y_emb = self.label_embeddings(y).unsqueeze(2)
        #x = torch.cat((z1, y_emb), dim=1)
        x = z1
        out = self.model(x)
        # TODO: Uncomment this
        # wfs = out[:, 0, :,]
        # lcn = torch.mean(out[:, 1, :], dim=1, keepdim=True)
        #return wfs, lcn
        return out


class CDiscriminator(nn.Module):
    r"""Conditional Discriminator Class for CGAN

    Args:
        num_variables (int): Number of variables.
        in_size (int, optional): Size of the input. Default is 1024.
        encoding_L (int, optional): Length of the encoding. Default is 8.
        in_channels (int, optional): Number of input channels. Default is 2.
        step_channels (int, optional): Number of channels in each step. Default is 32.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.
        nonlinearity (nn.Module, optional): Nonlinearity function to use. Default is None.
        last_nonlinearity (nn.Module, optional): Nonlinearity function to use in the last layer. Default is None.
        label_type (str, optional): The type of labels expected by the Discriminator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(
        self,
        num_variables,
        in_size=1024,
        encoding_L=4,
        in_channels=2,
        step_channels=32,
        batchnorm=True,
        nonlinearity=None,
        last_nonlinearity=None,
        label_type="required",
    ):
        super(CDiscriminator, self).__init__()
        self.input_dims = in_channels
        self.num_variables = num_variables
        self.label_embeddings = lambda x: torch.flatten(
            positional_encoding(x, encoding_L), start_dim=1
        )
        self.input_cond_dims = in_channels + 2 * encoding_L * num_variables  # + num_classes * encoding_L * 2
        self.label_type = label_type
        if in_size < 16 or ceil(log2(in_size)) != log2(in_size):
            raise Exception(
                "Input Image Size must be at least 16*16 and an exact power of 2"
            )
        

        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity
        d = self.n
        model = [
            nn.Sequential(nn.Conv1d(self.input_cond_dims, d, 4, 2, 1, bias=True), nl)
        ]
        if batchnorm is True:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.Conv1d(d, d * 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm1d(d * 2),
                        nl,
                    )
                )
                d *= 2
        else:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(nn.Conv1d(d, d * 2, 4, 2, 1, bias=use_bias), nl)
                )
                d *= 2
        self.disc = nn.Sequential(nn.Conv1d(d, 1, 4, 1, 0, bias=use_bias), last_nl)
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all disciminator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # def forward(self, x, norm, cond_vars):
    # TODO: Add normalization
    def forward(self, x, cond_vars):
        """
        Forward pass of the Conditional Generative Adversarial Network (CGAN) model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, waveform_size).
            norm (torch.Tensor): Normalization tensor of shape (batch_size, 1).
            cond_vars (torch.Tensor): Conditional variables tensor of shape (batch_size, num_cond_vars).
            feature_matching (bool, optional): Whether to return the feature matching output. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) if feature_matching is False, otherwise the feature matching output tensor.
        """
        x_ch = x.unsqueeze(1)

        # cond_emb = self.label_embeddings(cond_vars)
        # cond_emb = cond_emb.unsqueeze(2).expand(-1, cond_emb.size(1), x_ch.size(2))

        # norm_emb = norm.unsqueeze(2).expand(-1, 1, x_ch.size(2))

        # TODO: Undo this
        # x_ch = torch.cat((x_ch, cond_emb, norm_emb), dim=1)
        # x_ch = torch.cat((x_ch, cond_emb), dim=1)
        x_ch = self.model(x_ch)
        return self.disc(x_ch).view(-1, 1)
