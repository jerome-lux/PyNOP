import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union, Literal


class DeepONet(nn.Module):

    def __init__(
        self,
        branchnet=[100, 64, 64, 64, 64],
        trunknet=[2, 64, 64, 64, 64],
        encoder: Union[None, Callable] = None,
        decoder: Union[None, Callable] = None,
        activation: Union[None, Callable] = None,
        normalization: Union[None, Callable] = None,
        scale: Union[None, float, int, Literal["auto"]] = None,
    ):
        r"""
        Implement a general deeponet architecture (DON, Latent-DON, Fourier-MIONET, etc.).

        Arguments:
        - branchnet: either a neural network or a list of number of neurons by layer (MLP case)
        - trunknet: idem
        - encoder: None or an existing model. It encodes the input to the branchnet
        - decoder: None or an existing model. It decodes the ouput of the DON
        - activation: a valid activation function
        - normalization: a valid normalization layer or None (default DON has no normalization)
        - scale: the scaling factor applied to the output (can be 1/n or $1/ \sqrt(n) $ where n is the
          dimension of the output). If a number, it is used as a fixed factor; if 'auto', it becomes a
          learnable parameter (initialized to 1.0); if None, it is fixed to 1.

        Notes:
        The dimension of the input of the trunk net is the dimensionality N of the coordinates: [Bt, N]./ Bt is the number of query points.
        The input shape of the branch net depends on the chosen model:
        - MLP or Attention case: A tensor of shape [Bb, Neval, C] where Bb is the number of evaluated functions and Neval is the number of evaluations of the function(s).
        C can be > 1 if the function is a vector function.
        - CNN or FNO: a structured 2D or 3D grid [Bb, C, H, W] for example).
        Note that the batch dimension of the two networks may not be the same: we evaluate Bb functions at Bt coordinates.
        The output tensor is therefore [Bb, Bt, ...]: Bt predictions for each of the Bb functions. This is NOT equivalent to the multi-branch DeepONet
        If there is a decoder, this tensor is reshaped to [Bb*Bt, ...], because the decoder is generally trained to decode only one image at a time.
        """

        super(DeepONet, self).__init__()

        # scale: a number -> fixed factor, 'auto' -> learnable parameter (init 1.0), None -> fixed to 1
        if scale is None:
            self.scale = 1.0
        elif scale == "auto":
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale = float(scale)

        use_bias = normalization is None

        def _build_mlp(dims):
            """Build a callable MLP (nn.Sequential) from a list of layer widths, or return
            ``dims`` unchanged if it is already a callable module (e.g. CNN, FNO).

            No activation / normalization is applied after the last linear layer.
            """
            if callable(dims):
                return dims
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
                if i < len(dims) - 2:  # skip norm/activation on the final layer
                    if normalization is not None:
                        layers.append(normalization(dims[i + 1]))
                    if activation is not None:
                        layers.append(activation())
            return nn.Sequential(*layers)

        self.branchnet = _build_mlp(branchnet)
        self.trunknet = _build_mlp(trunknet)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, u, y):

        if self.encoder is not None:
            u = self.encoder(u)

        u = self.branchnet(u)
        y = self.trunknet(y)
        # The output of the branchnet can have a different shape to that of the trunknet,
        # but it **must** have the same number of channels
        x = torch.einsum("ji, ki...-> kj...", y, u) * self.scale

        if self.decoder is not None:
            # merge the two batch dimensions before decoding
            x = torch.flatten(x, start_dim=0, end_dim=1)
            x = self.decoder(x)

        return x
