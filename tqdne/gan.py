import os
import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdne.ganutils.evaluation import evaluate_model

torch.set_default_dtype(torch.float64)


def embed(in_chan, out_chan):
    """
    Creates embeding with 4 dense layers
    Progressively grows the number of output nodes
    """
    layers = nn.Sequential(
        nn.Linear(in_chan, 32),
        torch.nn.ReLU(),
        nn.Linear(32, 64),
        torch.nn.ReLU(),
        nn.Linear(64, 256),
        torch.nn.ReLU(),
        nn.Linear(256, 512),
        torch.nn.ReLU(),
        nn.Linear(512, 1024),
        torch.nn.ReLU(),
        # nn.Linear(1024, 2048), torch.nn.ReLU(),
        nn.Linear(1024, out_chan),
        torch.nn.ReLU(),
    )

    return layers


def FCNC(n_vs=150, hidden_1=256, hidden_2=512):
    """
    Fully connected neural net for normalization factors
    """
    layers = nn.Sequential(
        nn.Linear(n_vs, hidden_1),
        torch.nn.ReLU(),
        nn.Linear(hidden_1, hidden_2),
        torch.nn.ReLU(),
        nn.Linear(hidden_2, hidden_2),
        torch.nn.ReLU(),
        nn.Linear(hidden_2, hidden_1),
        torch.nn.ReLU(),
        nn.Linear(hidden_1, 1),
        torch.nn.Tanh(),
    )

    return layers


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #  get embeddings for the conditional variables
        self.embed1 = embed(1, 1000)
        self.embed2 = embed(1, 1000)

        # embedding for normalization constant
        self.nn_cnorm = embed(1, 1000)

        # first layer
        # concatenate conditional variables | (6, 4000, 1) out
        # (Chan,H,W)
        # (7,4000,1) input 2-D tensor input
        self.conv1 = nn.Conv2d(
            4,
            16,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        # (16,2000,1) out | apply F.leaky_relu
        self.conv1b = nn.Conv2d(
            16,
            16,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        # (16,2000,1) out | apply F.leaky_relu

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        # (32,1000,1) out | apply F.leaky_relu
        self.conv2b = nn.Conv2d(
            32,
            32,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        # (32,1000,1) out | apply F.leaky_relu

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        # (64,500,1) out | apply F.leaky_relu
        self.conv3b = nn.Conv2d(
            64,
            64,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        # (64,500,1) out | apply F.leaky_relu

        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        # (128,250,1) out | apply F.leaky_relu
        self.conv4b = nn.Conv2d(
            128,
            128,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        # (128,250,1) out | apply F.leaky_relu

        self.conv5 = nn.Conv2d(
            128,
            256,
            kernel_size=(32, 1),
            stride=(2, 1),
            padding=(15, 0),
        )
        # (256,125,1) out | apply F.leaky_relu
        self.conv5b = nn.Conv2d(
            256,
            256,
            kernel_size=(31, 1),
            stride=(1, 1),
            padding=(15, 0),
        )
        # (256,125,1) out | apply F.leaky_relu

        self.fc0 = nn.Linear(31, 110)  # 125, 110)
        # (256x110) out | apply F.leaky_relu
        self.fc1 = nn.Linear(110, 128)
        # (256x128) out | apply F.leaky_relu
        self.fc1b = nn.Linear(128, 100)
        # (256x100) out | apply F.leaky_relu | flatten op | (256*100)
        self.fc2 = nn.Linear(256 * 100, 1)
        # (1) out

    def forward(
        self,
        x,
        ln_cn,
        v1,
        v2,
    ):
        # print('--------------- Discriminator -------------')
        # conv2D + leaky relu activation

        #  apply embeddings for the conditional variables
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)

        # embedding normalization factors
        ln_cn = self.nn_cnorm(ln_cn)
        # print('nn_cnorm(ln_cn) shape:', ln_cn.shape)

        # # reshape
        v1 = v1.view(-1, 1, 1000, 1)
        v2 = v2.view(-1, 1, 1000, 1)
        ln_cn = ln_cn.view(-1, 1, 1000, 1)
        x = x.view(-1, 1, 1000, 1)

        # concatenate conditional variables to input
        x = torch.cat(
            [
                x,
                ln_cn,
                v1,
                v2,
            ],
            1,
        )
        # print('torch.cat([x, ln_cn, v1, v2, v3,]', x.shape)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv1 shape:', x.shape)

        x = self.conv1b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv1b shape:', x.shape)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv2 shape:', x.shape)

        x = self.conv2b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv2b shape:', x.shape)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv3 shape:', x.shape)

        x = self.conv3b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv3b shape:', x.shape)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv4 shape:', x.shape)

        x = self.conv4b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv4b shape:', x.shape)

        x = self.conv5(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv4 shape:', x.shape)

        x = self.conv5b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv4b shape:', x.shape)

        x = torch.squeeze(x, dim=3)
        # print('torch.squeeze shape:', x.shape)

        x = self.fc0(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc0 shape:', x.shape)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc1 shape:', x.shape)

        x = self.fc1b(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc1b shape:', x.shape)

        # flsatten to input into Last FC layer
        x = x.view(-1, 256 * 100)
        # print('view(-1, 256 * 100):', x.shape)

        out = self.fc2(x)
        # print('fc2 shape:', out.shape)

        return out


class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()

        # complete init function

        # first, fully-connected layer
        # input: (3,100) noise vector
        self.fc00 = nn.Linear(z_size, 150, bias=False)
        # output: (3,150)
        self.batchnorm00 = nn.BatchNorm1d(1)
        # output: (3,150) | apply expanddim  | (3,150,1) out

        #  ---- get embeddings for the conditional variables ----
        self.embed1 = embed(1, 150)
        self.embed2 = embed(1, 150)

        # ------------------------------------------------------

        # output after concatenating conditional variables: | (6, 150, 1) out

        self.conv0 = nn.Conv2d(
            3,
            6,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (6, 150, 1) | apply batchnorm
        self.batchnorm0 = nn.BatchNorm2d(6)
        # output: (6, 150, 1)

        self.conv0b = nn.Conv2d(
            6,
            6,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (6,150,1) | apply batchnorm
        self.batchnorm0b = nn.BatchNorm2d(6)
        # output: (6,150,1)

        self.conv0c = nn.Conv2d(
            6,
            3,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (3,150,1) | apply batchnorm
        self.batchnorm0c = nn.BatchNorm2d(3)
        # output: (3,150,1) | apply torch.squeeze | (3,150) out

        self.fc01 = nn.Linear(150, 250 + 50, bias=False)
        # output: (3,250)
        self.batchnorm01 = nn.BatchNorm1d(3)
        # output: (3,250) | apply reshape | (6, 125, 1) out

        self.resizenn1 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )
        # utput: (6, 250, 1)

        self.conv1 = nn.Conv2d(
            6,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (16, 250, 1)
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv1b = nn.Conv2d(
            16,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (16, 250, 1)
        self.batchnorm1b = nn.BatchNorm2d(16)

        self.resizenn2 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )
        # output: (16, 500, 1)

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (32, 500, 1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv2b = nn.Conv2d(
            32,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (32, 500, 1)
        self.batchnorm2b = nn.BatchNorm2d(32)

        # -- resize operation --
        self.resizenn3 = nn.Upsample(
            scale_factor=(2, 1),
            mode="nearest",
        )
        # output: (64, 1000, 1)

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 1000, 1)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv3b = nn.Conv2d(
            64,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 1000, 1)
        self.batchnorm3b = nn.BatchNorm2d(64)

        # -- resize operation --
        # self.resizenn4 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )
        # output: (64, 2000, 1)

        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 2000, 1)
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv4b = nn.Conv2d(
            128,
            128,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 2000, 1)
        self.batchnorm4b = nn.BatchNorm2d(128)

        # -- resize operation --
        # self.resizenn5 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )
        # output: (128, 4000, 1)

        self.conv5 = nn.Conv2d(
            128,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 4000, 1)
        self.batchnorm5 = nn.BatchNorm2d(64)

        self.conv5b = nn.Conv2d(
            64,
            64,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (64, 4000, 1)
        self.batchnorm5b = nn.BatchNorm2d(64)

        self.conv5c = nn.Conv2d(
            64,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (32, 4000, 1)
        self.batchnorm5c = nn.BatchNorm2d(32)

        self.conv5d = nn.Conv2d(
            32,
            32,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (16, 4000, 1)
        self.batchnorm5d = nn.BatchNorm2d(32)

        self.conv5e = nn.Conv2d(
            32,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (32, 4000, 1)
        self.batchnorm5e = nn.BatchNorm2d(16)

        self.conv5f = nn.Conv2d(
            16,
            16,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (16, 4000, 1)
        self.batchnorm5f = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(
            16,
            1,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0),
        )
        # output: (3, 4000, 1)

        # tanh activation function for generator output
        self.tanh4 = nn.Tanh()
        # output: (3, 4000, 1)

        # Module for normalization constant
        self.fc_lcn = FCNC(n_vs=150, hidden_1=256, hidden_2=512)
        # output: (3)

    def forward(
        self,
        x,
        v1,
        v2,
    ):
        # print('----------------- Generator -------------')
        # fully-connected + reshape
        # print('shape x init:', x.shape)

        x = self.fc00(x)

        x = self.batchnorm00(x)
        # print('fc00 shape:', x.shape)

        # expand dimension
        x = torch.unsqueeze(x, 3)
        # print('torch.unsqueeze shape:', x.shape)

        # ----------- Conditional variables  ------------
        #  apply embeddings for the conditional variables
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)

        v1 = v1.view(-1, 1, 150, 1)
        v2 = v2.view(-1, 1, 150, 1)

        # concatenate conditional variables to input
        # print("----------------------------------------------")
        # print(x.shape, v1.shape, v2.shape)
        # print("----------------------------------------------")
        x = torch.cat([x, v1, v2], 1)
        # print('torch.cat([x, v1, v2, v3],1) shape: ', x.shape)
        # ------------------------------------------------

        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = F.relu(x)
        # print('conv0 shape:', x.shape)

        x = self.conv0b(x)
        x = self.batchnorm0b(x)
        x = F.relu(x)
        # print('conv0b shape:', x.shape)

        x = self.conv0c(x)
        x = self.batchnorm0c(x)
        x = F.relu(x)
        # print('conv0c shape:', x.shape)

        x = torch.squeeze(x, 3)
        # print('torch.squeeze shape:', x.shape)

        x = self.fc01(x)
        # print('fc01 shape:', x.shape)
        x = self.batchnorm01(x)
        x = F.relu(x)
        # print('fc01 bnorm +relu shape:', x.shape)

        # ----------- get normalization constants ---------
        # get norm factor
        xcn = x[:, :, 250:]
        # print('x[:,:,250:] shape:', xcn.shape)
        # flatten to input into FC prediction layer
        xcn = xcn.reshape(-1, 3 * 50)
        # print('view(-1, 3 * 50):', xcn.shape)
        # get seed for waves
        x = x[:, :, :250]
        # # print('x[:,:,:250] shape:', x.shape)

        # --------------------------------------------
        x = x.reshape(-1, 6, 125, 1)
        # print('x.reshape(-1, 6, 125, 1) shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn1(x)
        # print('resizenn1 shape:', x.shape)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        # print('conv1 shape:', x.shape)

        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = F.relu(x)
        # print('conv1b shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn2(x)
        # print('resizenn2 shape:', x.shape)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        # print('conv2 shape:', x.shape)

        x = self.conv2b(x)
        x = self.batchnorm2b(x)
        x = F.relu(x)
        # print('cconv2b shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn3(x)
        # print('resizenn3 shape:', x.shape)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        # print('conv3 shape:', x.shape)

        x = self.conv3b(x)
        x = self.batchnorm3b(x)
        x = F.relu(x)
        # print('cconv3b shape:', x.shape)

        # resize_nearest_neighbor
        # x = self.resizenn4(x)
        # print('resizenn4 shape:', x.shape)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        # print('conv4 shape:', x.shape)

        x = self.conv4b(x)
        x = self.batchnorm4b(x)
        x = F.relu(x)
        # print('conv4b shape:', x.shape)

        # resize_nearest_neighbor
        # x = self.resizenn5(x)
        # print('resizenn5 shape:', x.shape)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        # print('conv5 shape:', x.shape)

        x = self.conv5b(x)
        x = self.batchnorm5b(x)
        x = F.relu(x)
        # print('cconv5b shape:', x.shape)

        x = self.conv5c(x)
        x = self.batchnorm5c(x)
        x = F.relu(x)
        # print('cconv5c shape:', x.shape)

        x = self.conv5d(x)
        x = self.batchnorm5d(x)
        x = F.relu(x)
        # print('cconv5d shape:', x.shape)

        x = self.conv5e(x)
        x = self.batchnorm5e(x)
        x = F.relu(x)
        # print('cconv5e shape:', x.shape)

        x = self.conv5f(x)
        x = self.batchnorm5f(x)
        x = F.relu(x)
        # print('cconv5f shape:', x.shape)

        # ----

        x = self.conv6(x)
        # print('conv6 shape:', x.shape)

        # final outpsut
        x_out = self.tanh4(x)
        # print('tanh4 shape:', x.shape)

        # normalization output
        xcn_out = self.fc_lcn(xcn)
        # print('self.fc_lcn shape:', xcn_out.shape)

        return (x_out, xcn_out)


class LogGanCallback(L.callbacks.Callback):
    def __init__(self, mlf_logger, dataset, every=5) -> None:
        super().__init__()
        self.mlf_logger = mlf_logger
        self.scores = []
        self.scores_val = []
        self.dataset = dataset
        # self.dataset_train = dataset_train
        self.cur_epoch = 0
        self.every = every

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
    #     print(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        # if (pl_module.cur_epoch + 1) % 10 != 0:
        #     return
        self.cur_epoch += 1
        save_loc_epoch = f"{self.mlf_logger.save_dir}/{self.mlf_logger.name}/{self.mlf_logger.run_id}/model_epoch_{self.cur_epoch}"
        print(save_loc_epoch)
        # self.mlf_logger.experiment.pytorch.save_model(self.G, save_loc_epoch)
        # self.mlf_logger.experiment.pytorch.log_model(self.G, save_loc_epoch)

        metrics_dir = os.path.join(save_loc_epoch, "metrics")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        grid_dir = os.path.join(save_loc_epoch, "grid_plots")
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)

        fig_dir = os.path.join(save_loc_epoch, "figs")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        epoch_loc_dirs = {
            "output_dir": save_loc_epoch,
            "metrics_dir": metrics_dir,
            "grid_dir": grid_dir,
            "fig_dir": fig_dir,
        }

        n_waveforms = 72 * 5
        evaluate_model(
            pl_module.G,
            n_waveforms,
            self.dataset,
            epoch_loc_dirs,
            self.mlf_logger,
            pl_module.hparams,
        )


class GAN(L.LightningModule):
    def __init__(
        self,
        reg_lambda,
        latent_dim,
        n_critics,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs,
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
        self.G = Generator(z_size=self.hparams.latent_dim).to(self.device)
        self.D = Discriminator().to(self.device)

    def random_z(self) -> torch.TensorType:
        mean = torch.zeros(self.hparams.batch_size, 1, self.hparams.latent_dim)
        return torch.normal(mean=mean).to(self.device)
        # z = np.random.normal(size=[self.hparams.batch_size, 1, self.hparams.latent_dim]).astype(dtype=np.float64)
        # z = torch.from_numpy(z)
        # return z

    def forward(self, z, v1, v2):
        return self.G(z, v1, v2)

    def discriminator_loss(self, real_wfs, real_lcn, fake_wfs, fake_lcn, i_vc):
        Nsamp = real_wfs.size(0)
        # random constant
        alpha = torch.Tensor(np.random.random((Nsamp, 1, 1, 1)))
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
        )
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
        )
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
        g_loss = -torch.mean(self.D(wfs, lcn, *i_vg)).to(self.device)
        self.g_loss = g_loss
        return g_loss

    def training_step(self, batch, batch_idx):
        real_wfs, real_lcn, i_vc = batch
        real_wfs = real_wfs.to(self.device)
        real_lcn = real_lcn.to(self.device)
        i_vc = [i.to(self.device) for i in i_vc]
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
        d_loss = self.discriminator_loss(real_wfs, real_lcn, fake_wfs, fake_lcn, i_vc)
        self.log("d_val_wloss", self.d_wloss, prog_bar=True, on_epoch=True)
        self.log("d_val_gploss", self.d_gploss, prog_bar=True, on_epoch=True)

        # get random sampling of conditional variables
        (fake_wfs, fake_lcn) = self.G(self.random_z(), *i_vc)

        # calculate loss
        g_loss = self.generator_loss(fake_wfs, fake_lcn, i_vc)
        self.log("g_val_loss", g_loss, prog_bar=True, on_epoch=True)
