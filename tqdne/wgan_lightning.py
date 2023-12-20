import pytorch_lightning as L
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdne.models.wgan import WGenerator, WDiscriminator

torch.set_default_dtype(torch.float64)

class WGAN(L.LightningModule):
    def __init__(
        self,
        waveform_size,
        reg_lambda,
        n_critics,
        optimizer_params,
        generator_params,
        discriminator_params,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.gp_weight = reg_lambda
        self.critic_iterations = n_critics
        self.print_every = 5
        # networks
        #TODO: Change num_variables to 2, out_channels to 2
        self.G = WGenerator(**generator_params)
        self.D = WDiscriminator(**discriminator_params)

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples)).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data
    
    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :]

    def forward(self, z, cond_tensor: torch.TensorType):
        return self.G(z, cond_tensor)
    
    def generator_step(self, data):
        self.toggle_optimizer(self.g_opt)
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        self.g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.g_opt.step()
        self.log("g_loss", g_loss)
        self.untoggle_optimizer(self.g_opt)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def discriminator_step(self, data):
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        data = Variable(data).to(self.device)
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.log("GP", gradient_penalty)

        self.toggle_optimizer(self.d_opt)
        self.d_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        self.manual_backward(d_loss)
        self.d_opt.step()
        self.log("d_loss", d_loss)
        self.untoggle_optimizer(self.d_opt)

    def training_step(self, batch, batch_idx):
        self.g_opt, self.d_opt = self.optimizers()
        data = batch
        self.discriminator_step(data)
        ### -------------- TAKE GENERATOR STEP ------------------------
        if (batch_idx + 1) % self.critic_iterations == 0:
            self.generator_step(data)

    def configure_optimizers(self):
        lr = self.hparams.optimizer_params["lr"]
        b1 = self.hparams.optimizer_params["b1"]
        b2 = self.hparams.optimizer_params["b2"]

        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []