import torch
from tqdne.models.model import MLP


def test_mlp():
    batch_size = 64

    hidden_dim = 256
    num_classes = 10
    n_layers = 3
    input_dim = 28 * 28

    net = MLP(input_dim, hidden_dim, num_classes, n_layers, use_softmax=True)

    x = torch.randn(batch_size, input_dim)
    y = net(x)
    assert y.shape == (batch_size, num_classes)
    assert y.max() <= 1
    assert y.min() >= 0
