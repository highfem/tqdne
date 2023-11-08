import pytorch_lightning as L
import numpy as np
import torch
from tqdne.models.gan import Generator, Discriminator

torch.set_default_dtype(torch.float64)


class GAN(L.LightningModule):
    def __init__(
        self,
        reg_lambda,
        latent_dim,
        n_critics,
        batch_size,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.cur_epoch = 0
        self.d_loss = None
        self.d_wloss = None
        self.d_gploss = None

        self.g_loss = None
        self.g_val_loss = None

        # networks
        self.G = Generator(z_size=self.hparams.latent_dim)
        self.D = Discriminator()

    def random_z(self) -> torch.TensorType:
        mean = torch.zeros(self.hparams.batch_size, 1, self.hparams.latent_dim)
        return torch.normal(mean=mean).to(self.device)
        # z = np.random.normal(size=[self.hparams.batch_size, 1, self.hparams.latent_dim]).astype(dtype=np.float64)
        # z = torch.from_numpy(z)
        # return z

    def sample(self, v1, v2):
        return self.forward(self.random_z(), v1, v2)

    def forward(self, z, v1, v2):
        return self.G(z, v1, v2)

    def discriminator_loss(self, real_wfs, real_lcn, fake_wfs, fake_lcn, i_vc):
        Nsamp = real_wfs.size(0)
        # random constant
        alpha = torch.Tensor(np.random.random((Nsamp, 1, 1, 1))).type_as(real_wfs)
        alpha_cn = alpha.view(Nsamp, 1)
        # Get random interpolation between real and fake samples
        real_wfs = real_wfs.view(-1, 1, 1000, 1)
        real_lcn = real_lcn.view(real_lcn.size(0), -1)

        # for waves
        Xwf_p = (alpha * real_wfs + ((1.0 - alpha) * fake_wfs)).requires_grad_(True)
        # for normalization
        Xcn_p = (alpha_cn * real_lcn + ((1.0 - alpha_cn) * fake_lcn)).requires_grad_(
            True
        )
        # apply dicriminator
        D_xp = self.D(Xwf_p, Xcn_p, *i_vc)
        # Get gradient w.r.t. interpolates waveforms
        Xout_wf = torch.autograd.Variable(
            torch.Tensor(Nsamp, 1).fill_(1.0), requires_grad=False
        ).type_as(D_xp)
        grads_wf = torch.autograd.grad(
            outputs=D_xp,
            inputs=Xwf_p,
            grad_outputs=Xout_wf,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads_wf = grads_wf.view(grads_wf.size(0), -1)
        # get gradients w.r.t. normalizations
        Xout_cn = torch.autograd.Variable(
            torch.Tensor(Nsamp, 1).fill_(1.0), requires_grad=False
        ).type_as(D_xp)
        grads_cn = torch.autograd.grad(
            outputs=D_xp,
            inputs=Xcn_p,
            grad_outputs=Xout_cn,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # concatenate grad vectors
        grads = torch.cat([grads_wf, grads_cn], 1)

        y_hat = self.D(real_wfs, real_lcn, *i_vc)
        y = self.D(fake_wfs, fake_lcn, *i_vc)

        self.d_gploss = (
            self.hparams.reg_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        )
        self.d_wloss = -torch.mean(y_hat) + torch.mean(y)
        self.d_loss = self.d_wloss + self.d_gploss

        return self.d_loss

    def generator_loss(self, wfs, lcn, i_vg):
        g_loss = -torch.mean(self.D(wfs, lcn, *i_vg))
        self.g_loss = g_loss
        return g_loss

    def training_step(self, batch, batch_idx):
        real_wfs, real_lcn, i_vc = batch
        print(real_wfs.shape, real_lcn.shape, i_vc[0].shape, i_vc[1].shape)
        # real_wfs = real_wfs.to(self.device)
        # real_lcn = real_lcn.to(self.device)
        # i_vc = [i.to(self.device) for i in i_vc]
        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_d)

        ### ---------- DISCRIMINATOR STEP ---------------
        optimizer_d.zero_grad()
        (fake_wfs, fake_lcn) = self.G(self.random_z(), *i_vc)

        d_loss = self.discriminator_loss(real_wfs, real_lcn, fake_wfs, fake_lcn, i_vc)
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.log("d_train_gploss", self.d_gploss, prog_bar=True, on_epoch=True)
        self.log("d_train_wloss", self.d_wloss, prog_bar=True, on_epoch=True)
        self.log("d_train_loss", self.d_loss, prog_bar=True, on_epoch=True)
        self.untoggle_optimizer(optimizer_d)
        ### ---------- END DISCRIMINATOR STEP ---------------

        ### -------------- TAKE GENERATOR STEP ------------------------
        if (batch_idx + 1) % self.hparams.n_critics == 0:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()

            (fake_wfs, fake_lcn) = self.G(self.random_z(), *i_vc)
            g_loss = self.generator_loss(fake_wfs, fake_lcn, i_vc)
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            self.log("g_train_loss", g_loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def validation_step(self, batch, batch_idx):
        ### ---------- DISCRIMINATOR STEP ---------------
        # 1. get real data
        # get random sample
        real_wfs, real_lcn, i_vc = batch

        # generate a batch of waveform no autograd
        (fake_wfs, fake_lcn) = self.G(self.random_z(), *i_vc)

        # random constant
        with torch.enable_grad():
            d_loss = self.discriminator_loss(
                real_wfs, real_lcn, fake_wfs, fake_lcn, i_vc
            )
        self.log("d_val_wloss", self.d_wloss, prog_bar=True, on_epoch=True)
        self.log("d_val_gploss", self.d_gploss, prog_bar=True, on_epoch=True)

        # get random sampling of conditional variables
        (fake_wfs, fake_lcn) = self.G(self.random_z(), *i_vc)

        # calculate loss
        g_loss = self.generator_loss(fake_wfs, fake_lcn, i_vc)
        self.log("g_val_loss", g_loss, prog_bar=True, on_epoch=True)
