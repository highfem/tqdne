import pytorch_lightning as L
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdne.wgan_model import WGenerator, WDiscriminator


class WGAN(L.LightningModule):
    def __init__(
        self,
        reg_lambda,
        n_critics,
        optimizer_params,
        generator_params,
        discriminator_params,
        conditional=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.conditional = conditional
        self.num_steps = 0
        self.gp_weight = reg_lambda
        self.critic_iterations = n_critics
        self.print_every = 5
        # networks
        self.G = WGenerator(**generator_params)
        self.D = WDiscriminator(**discriminator_params)

    def sample_generator(self, num_samples, cond=None):
        latent_samples = Variable(self.G.sample_latent(num_samples)).to(self.device)
        generated_data = self.G(latent_samples, cond)
        return generated_data
    
    def sample(self, num_samples, cond=None):
        generated_data = self.sample_generator(num_samples, cond)
        # Remove color channel
        return generated_data.data.detach().cpu().numpy()
    
    def evaluate(self, batch):
        shape = batch["high_res"].shape[0]
        cond = batch["cond"]
        sample = self.sample(shape, cond)
        return {"high_res": sample}
    
    def generator_step(self, data, cond=None):
        self.toggle_optimizer(self.g_opt)
        batch_size = data.size()[0]
        # Generate a sample
        generated_data = self.sample_generator(batch_size, cond)
        # Evaluate the sample
        d_generated = self.D(generated_data, cond)
        g_loss = - d_generated.mean()
        self.g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.g_opt.step()
        self.log("g_loss", g_loss)
        self.untoggle_optimizer(self.g_opt)

    def _gradient_penalty(self, real_data, generated_data, cond):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, cond)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, len),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def discriminator_step(self, data, cond):
        batch_size = data.size()[0]
        
        # Generate a sample
        generated_data = self.sample_generator(batch_size, cond)
        data = Variable(data).to(self.device)
        # Evaluate the sample and the real data
        d_real = self.D(data, cond)
        d_generated = self.D(generated_data, cond)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data, cond)
        self.log("GP", gradient_penalty, prog_bar=True)

        self.toggle_optimizer(self.d_opt)
        self.d_opt.zero_grad()
        d_critic =  d_generated.mean() - d_real.mean()
        self.log("negative-critic", -d_critic, prog_bar=True)
        d_loss = d_critic + gradient_penalty
        self.manual_backward(d_loss)
        self.d_opt.step()
        self.log("d_loss", d_loss, prog_bar=True)
        self.untoggle_optimizer(self.d_opt)

    def training_step(self, batch, batch_idx):
        self.g_opt, self.d_opt = self.optimizers()
        if self.conditional:
            data, cond = batch["high_res"], batch["cond"]
        else:
            data, cond = batch["high_res"], None
        self.discriminator_step(data, cond)
        if (batch_idx + 1) % self.critic_iterations == 0:
            self.generator_step(data, cond)
    
    def validation_step(self, batch, batch_idx):
        if self.conditional:
            data, cond = batch["high_res"], batch["cond"]
        else:
            data, cond = batch["high_res"], None
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, cond)
        data = Variable(data).to(self.device)
        d_real = self.D(data, cond)
        d_generated = self.D(generated_data, cond)
        d_critic =  d_generated.mean() - d_real.mean()
        with torch.enable_grad():
            gradient_penalty = self._gradient_penalty(data, generated_data, cond)
        self.log("val_GP", gradient_penalty)
        self.log("val_negative-critic", -d_critic)
        self.log("val_loss", -d_generated.mean())
    

    def configure_optimizers(self):
        lr = self.hparams.optimizer_params["lr"]
        b1 = self.hparams.optimizer_params["b1"]
        b2 = self.hparams.optimizer_params["b2"]

        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []