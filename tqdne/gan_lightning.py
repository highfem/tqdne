from flask import g
import pytorch_lightning as L
import numpy as np
import torch
from tqdne.models.cgan import CGenerator, CDiscriminator

torch.set_default_dtype(torch.float64)


class GAN(L.LightningModule):
    def __init__(
        self,
        waveform_size,
        reg_lambda,
        n_critics,
        batch_size,
        optimizer_params,
        generator_params,
        discriminator_params,
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
        #TODO: Change num_variables to 2, out_channels to 2
        self.G = CGenerator(**generator_params)
        self.D = CDiscriminator(**discriminator_params)

    def sample(self, num_waveforms, cond_list):
        z, cond_tensor = self.G.sampler(num_waveforms, cond_list, self.device)
        # TODO: Undo this
        w = self.forward(z, cond_tensor)
        w = w.view(w.size(0), -1)
        return w, None

    def forward(self, z, cond_tensor: torch.TensorType):
        return self.G(z, cond_tensor)
    
    def gradient_penalty_loss(self, real_wfs, real_lcn, fake_wfs, fake_lcn, cond_tensor):
        Nsamp = real_wfs.size(0)
        alpha = torch.Tensor(np.random.random((Nsamp, 1))).type_as(real_wfs)
        # print("real_wfs shape:", real_wfs.shape)
        # print("fake_wfs shape:", fake_wfs.shape)
        Xwf_p = (alpha * real_wfs + ((1.0 - alpha) * fake_wfs)).requires_grad_(True)
        # for normalization
        # Xcn_p = (alpha * real_lcn + ((1.0 - alpha) * fake_lcn)).requires_grad_(True)
        # apply dicriminator
        # D_xp = self.D(Xwf_p, Xcn_p, cond_tensor)
        D_xp = self.D(Xwf_p, cond_tensor)
        # Get gradient w.r.t. interpolates waveforms
        grads_wf = torch.autograd.grad(
            outputs=D_xp,
            inputs=Xwf_p,
            grad_outputs=torch.ones((Nsamp, 1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads_wf = grads_wf.view(grads_wf.size(0), -1)
        # TODO: Uncomment this
        # get gradients w.r.t. normalizations
        # grads_cn = torch.autograd.grad(
        #     outputs=D_xp,
        #     inputs=Xcn_p,
        #     grad_outputs=torch.ones((Nsamp, 1)).to(self.device),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # grads_cn = grads_cn.view(grads_cn.size(0), -1)
        # concatenate grad vectors
        # grads = torch.cat([grads_wf, grads_cn], 1)
        grads = grads_wf
        loss = (
            self.hparams.reg_lambda * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        )
        return loss
    
    def discriminator_wloss(self, real_wfs, real_lcn, fake_wfs, fake_lcn, cond_tensor):
        # y_real_sample = self.D(real_wfs, real_lcn, cond_tensor)
        # y_fake_sample = self.D(fake_wfs, fake_lcn, cond_tensor)
        y_real_sample = self.D(real_wfs, cond_tensor)
        y_fake_sample = self.D(fake_wfs, cond_tensor)
        loss = -torch.mean(y_real_sample) + torch.mean(y_fake_sample)
        return loss

    def discriminator_loss(self, real_wfs, real_lcn, cond_tensor):
        # print("cond_tensor: ", cond_tensor.shape)
        (fake_wfs, fake_lcn) = self.sample(self.hparams.batch_size, cond_tensor)
        self.d_gploss = self.gradient_penalty_loss(real_wfs, real_lcn, fake_wfs, fake_lcn, cond_tensor)
        self.d_wloss = self.discriminator_wloss(real_wfs, real_lcn, fake_wfs, fake_lcn, cond_tensor)
        self.d_loss = self.d_wloss + self.d_gploss
        return self.d_loss

    def generator_loss(self, wfs, lcn, i_vg):
        # g_loss = -torch.mean(self.D(wfs, lcn, i_vg))
        g_loss = -torch.mean(self.D(wfs, i_vg))
        self.g_loss = g_loss
        return g_loss

    def training_step(self, batch, batch_idx):
        real_wfs, real_lcn, i_vc = batch
        optimizer_g, optimizer_d = self.optimizers()
        
        ### ---------- DISCRIMINATOR STEP ---------------
        self.toggle_optimizer(optimizer_d)
        d_loss = self.discriminator_loss(real_wfs, real_lcn, i_vc)
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.log("d_train_gploss", self.d_gploss)
        self.log("d_train_wloss", self.d_wloss)
        self.log("d_train_loss", self.d_loss)
        self.untoggle_optimizer(optimizer_d)

        ### -------------- TAKE GENERATOR STEP ------------------------
        if (batch_idx + 1) % self.hparams.n_critics == 0:
            self.toggle_optimizer(optimizer_g)
            (fake_wfs, fake_lcn) = self.sample(self.hparams.batch_size, i_vc)
            g_loss = self.generator_loss(fake_wfs, fake_lcn, i_vc)
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            self.log("g_train_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.optimizer_params["lr"]
        momentum = self.hparams.optimizer_params["momentum"]
        # b1 = self.hparams.optimizer_params["b1"]
        # b2 = self.hparams.optimizer_params["b2"]

        # opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        # opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))    
        opt_g = torch.optim.SGD(self.G.parameters(), lr=lr, momentum=momentum)
        opt_d = torch.optim.SGD(self.D.parameters(), lr=lr, momentum=momentum)
        return [opt_g, opt_d], []

    def validation_step(self, batch, batch_idx):
        ### ---------- DISCRIMINATOR STEP ---------------
        # 1. get real data
        # get random sample
        real_wfs, real_lcn, i_vc = batch

        # print("i_vc shape:", i_vc.shape)
        with torch.enable_grad():
            d_loss = self.discriminator_loss(
                real_wfs, real_lcn, i_vc
            )
        self.log("d_val_wloss", self.d_wloss, on_step=True)
        self.log("d_val_gploss", self.d_gploss, on_step=True)

        # get random sampling of conditional variables
        (fake_wfs, fake_lcn) = self.sample(self.hparams.batch_size, i_vc)
        g_loss = self.generator_loss(fake_wfs, fake_lcn, i_vc)
        self.log("val_loss", g_loss, on_step=True)