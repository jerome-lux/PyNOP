import torch
import torch.nn as nn
import torch.nn.functional as F
from pynop.core.ops import CartesianEmbedding, SinusoidalEmbedding
from typing import Callable, Union

# [Modified] Forward path: encode -> branch/trunk -> combine (elementwise) -> optional decode (Aug 2025)
class DeepONet(nn.Module):

    def __init__(
        self,
        branchnet=[100, 64, 64, 64, 64],
        trunknet=[2, 64, 64, 64, 64],
        encoder: Union[None, Callable] = None,
        decoder: Union[None, Callable] = None,
        activation: Union[None, Callable] = None,
        normalization: Union[None, Callable] = None,
        scale: Union[None, float, int] = None,
    ):
        """
        Implement a general deeponet architecture (DON, Latent-DON, Fourier-MIONET, etc.).

        Arguments:
        - branchnet: either a neural network or a list of number of neurons by layer (MLP case)
        - trunknet: idem
        - encoder: None or an existing model. It encodes the input to the branchnet
        - decoder: None or an existing model. It decodes the ouput of the DON
        - activation: a valid activation function
        - normalization: a valid normalization layer or None (default DON has no normalization)
        - scale: the scaling factor applied to the output (can be 1/n or $1/ \\sqrt(n) $ where n is the dimension of the output)

        Notes:
        The dimension of the input of the trunk net is the dimensionality N of the coordinates: [Bt, N]./ Bt is the number of query points.
        The input shape of the branch net depends on the chosen model:
        - MLP case: A tensor of shape [Bb, Neval, C] where Bb is the number of evaluated functions and Neval is the number of evaluations of the function(s).
        C can be > 1 if the function is a vector function.
        - CNN or FNO: a structured 2D or 3D grid [Bb, C, H, W] for example).
        Note that the batch dimension of the two networks may not be the same: we evaluate Bb functions at Bt coordinates.
        The output tensor is therefore [Bb, Bt, ...]: Bt predictions for each of the Bb functions. This is NOT equivalent to the multi-branch DeepONet
        If there is a decoder, this tensor is reshaped to [Bb*Bt, ...], because the decoder is generally trained to decode only one image at a time.
        """

        super(DeepONet, self).__init__()

        self.activation = activation
        self.normalization = normalization
        self.scale = 1 if scale is None else nn.Parameter(torch.tensor(scale))
       
        
        use_bias = True if normalization is None else False

        if not callable(branchnet):
            # If not given, the branchnet is just a MLP
            self.branchlist = nn.ModuleList()
            for i, n in enumerate(branchnet[1:]):
                self.branchlist.append(nn.Linear(branchnet[i], n, bias=use_bias))
                if normalization is not None:
                    self.branchlist.append(self.normalization(n))
                if activation is not None:
                    self.branchlist.append(self.activation())
            # Last layer without activation
            self.branchlist.append(nn.Linear(branchnet[i + 1], branchnet[i + 2]))
        else:
            self.branchnet = branchnet

        # self.trunknet = nn.ModuleList()
        if not callable(trunknet):
            # If not given, the branchnet is just a MLP
            for i in range(len(trunknet) - 1):
                self.trunknet.append(nn.Linear(trunknet[i], trunknet[i + 1], bias=use_bias))
                if i < len(trunknet) - 2:
                    if normalization is not None:
                        self.trunknet.append(normalization(trunknet[i + 1]))
                    if activation is not None:
                        self.trunknet.append(activation())


                #---------------------------------------OLD VERSION---------------------------------#
            # for i, n in enumerate(trunknet[1:]):
            #     self.trunknet.append(nn.Linear(trunknet[i], n, bias=use_bias))
            #     if normalization is not None:
            #         self.trunknet.append(self.normalization(n))
            #     if activation is not None:
            #         self.trunknet.append(self.activation())
            # Last layer without activation
            # self.trunknet.append(nn.Linear(trunknet[i + 1], trunknet[i + 2]))
        else:
            self.trunknet = trunknet

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, u, y):

        if self.encoder is not None:
            u = self.encoder(u)

        u = self.branchnet(u)
        y = self.trunknet(y)

        # x = torch.einsum("bi, tj -> bt", u, y) * self.scale
        # x = torch.einsum("ji, ki...-> kj...", y, u) * self.scale
        # x = torch.einsum("bi, bi -> b", u, y) 
        # x = x * self.scale
        latent = u * y

        if self.decoder is not None:
            # merge the two batch dimensions before decoding
            # x = torch.flatten(x, start_dim=0, end_dim=1)
            # x = x.view(-1, 1)                                   
            # x = x.expand(-1, 2048)                        
            x = latent.view(-1, 32, 8, 8)         
            x = self.decoder(x)
        return x
