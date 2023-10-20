from torch import nn
import torch

class MLP(nn.Module):
    """Multi-layer perceptron model.

    It uses leaky ReLU as activation function and the last layer is a
    linear layer.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    hidden_dim : int
        Dimension of the hidden layers.
    output_dim : int
        Dimension of the output.
    n_layers : int
        Number of hidden layers.
    use_softmax : bool
        Whether to use softmax as the activation function of the last layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.use_softmax = use_softmax
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.functional.leaky_relu(x)
        x = self.layers[-1](x)
        if self.use_softmax:
            x = nn.functional.softmax(x, dim=1)
        return x




# class CNN1D(nn.Module):
#     """ Fully convolutional neural network for 1D data.
    
#     Parameters
#     ----------
#     input_channel : int
#         Number of input channels.
#     n_convs : list of int
#         Number of convolutions per layer.
#     kernel_sizes : list of int
#         Kernel sizes per layer.
#     strides : list of int
#         Strides per layer.
#     """
#     def __init__(self, input_channel, n_convs, kernel_sizes, strides):

#         super(CNN1D, self).__init__()
#         self.input_channel = input_channel
#         self.n_convs = n_convs
#         self.strides = strides
#         self.activation = nn.LeakyReLU()

#         self.kernel_sizes = kernel_sizes
#         self.conv = nn.ModuleList()

#         # Append a list of convolutional neural network, starting with the number of input channels specified in the model.
#         nconv_old = self.input_channel
#         for n_conv, kernel_size, stride in zip(self.n_convs, self.kernel_sizes, self.strides):
#             self.conv.append(nn.Conv1d(in_channels=nconv_old, 
#                                        out_channels=n_conv, 
#                                        kernel_size=kernel_size, 
#                                        stride=stride, 
#                                        padding=kernel_size//2))
#             nconv_old = n_conv

        


#     def forward(self, x):
#         squeeze = False
#         if len(x.shape) == 2:
#             x = x.unsqueeze(0)
#             squeeze = True
#         assert len(x.shape) == 3

#         # do convolution
#         n = len(self.conv)
#         for i, module in enumerate(self.conv):
#             x = module(x)
#             if i < n - 1:
#                 x = self.activation(x)

#         if squeeze:
#             x = x.squeeze(0)

#         return x

class CNN2D(nn.Module):
    """ Fully convolutional neural network for 2D data with global average pooling.

    Parameters
    ----------
    input_channel : int
        Number of input channels.
    n_convs : list of int
        Number of convolutions per layer.
    kernel_sizes : list of int
        Kernel sizes per layer.
    strides : list of int
        Strides per layer.
    """
    
    def __init__(self, input_channel, convs, kernel_sizes, strides):

        super(CNN2D, self).__init__()
        self.input_channel = input_channel
        self.convs = convs
        self.strides = strides
        self.activation = nn.LeakyReLU()

        self.kernel_sizes = kernel_sizes
        self.conv = nn.ModuleList()

        # Append a list of convolutional neural network, starting with the number of input channels specified in the model.
        nconv_old = self.input_channel
        for n_conv, kernel_size, stride in zip(self.convs, self.kernel_sizes, self.strides):
            self.conv.append(nn.Conv2d(in_channels=nconv_old, 
                                       out_channels=n_conv, 
                                       kernel_size=kernel_size, 
                                       stride=stride, 
                                       padding=kernel_size//2))
            nconv_old = n_conv


    def forward(self, x):
        squeeze = False
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            squeeze = True
        assert len(x.shape) == 4

        # do convolution
        n = len(self.conv)
        for i, module in enumerate(self.conv):
            x = module(x)
            if i < n - 1:
                x = self.activation(x)
        
        # Average pooling over the pixels
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = torch.mean(x, dim=2)
        if squeeze:
            x = x.squeeze(0)

        return x