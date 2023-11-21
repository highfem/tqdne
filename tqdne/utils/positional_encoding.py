import torch
import numpy as np

def positional_encoding(x, L, xrange=1.0):
    """Compute the positional encoding for a given vector x

    The positional encoding is a vector of length 2L, where L is the number of frequencies to use.
    The first L entries are the sine of the input vector multiplied by $2^\ell * \pi * x$, where $\ell$ is the index of the frequency.
    The second L entries are the cosine of the input vector multiplied by $2^\ell * \pi * x$, where $\ell$ is the index of the frequency.

    Parameters
    ----------
    x : torch.Tensor
        Input vector
    L : int
        Number of frequencies to use
    
    Returns
    -------
    torch.Tensor
        The positional encoding of x. The shape is the same as x, with an additional dimension of size 2L appended to the end.
    """


    # assert len(x.shape)==1
    # n = len(x)
    # x = np.expand_dims(x,1)
    # ell = np.arange(L)
    # fac = 2**ell
    # fac = np.expand_dims(fac, 0)
    # inner = fac*np.pi * x
    # assert inner.shape == (n, L)
    # s = np.sin(inner)
    # c = np.cos(inner)

    # out = np.concatenate([s,c], axis=1)
    # assert out.shape == (n, 2*L)
    # return out

    isnumpy = type(x) == np.ndarray
    if isnumpy:
        x = torch.from_numpy(x)
    x /= (xrange *2) 
    shape = x.shape
    ndim = len(shape)
    x = x.unsqueeze(ndim)
    ell = torch.arange(L, requires_grad=False, dtype=x.dtype, device=x.device)
    fac = 2**ell
    fac = fac.view(*([1]*ndim),L)
    inner = fac*np.pi * x
    assert inner.shape == (*shape, L)
    s = torch.sin(inner)
    c = torch.cos(inner)
    out = torch.cat([s,c], dim=ndim)
    assert out.shape == (*shape, 2*L)
    if isnumpy:
        out = out.numpy()
    return out