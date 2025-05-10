from typing import Callable, Union, Sequence
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from pynop.core.blocks import CoDABlock2D


class CoDANO(nn.Module):
    """Implementation of Co-domain attention model as described in [1]_.
    This code is a simplified version of that found in https://github.com/neuraloperator/neuraloperator

    References
    ----------
    .. [1]: M. Rahman, R. George, M. Elleithy, D. Leibovici, Z. Li, B. Bonev,
        C. White, J. Berner, R. Yeh, J. Kossaifi, K. Azizzadenesheli, A. Anandkumar (2024).
        "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs."
        arxiv:2403.12553
    """

    def __init__(
        self,
        variable_ids: Sequence,
        hidden_lifting_channels: int = 64,
        hidden_variable_codimension: int = 32,
        positional_encoding_dim: int = 8,
        positional_encoding_modes: Sequence = (16, 16),
        static_channel_dim: int = 0,
        modes: Union[Sequence, int] = (16, 16),
        n_layers: int = 4,
        n_heads: int = 1,
        per_channel_attention: bool = False,
        activation: Callable = nn.GELU,
        norm: Callable = partial(nn.InstanceNorm2d, affine=True),
        ndim: int = 2,
        spectral_compression_factor: Sequence = (2, 2, 2),
    ):

        super().__init__()

        self.n_layers = n_layers
        self.activation = activation
        self.ndim = ndim
        self.variable_ids = variable_ids
        self.hidden_variable_codimension = hidden_variable_codimension

        # Create the params for positionnal encoding
        # Here we could also use a tucker decomposition to decrease the number of params
        self.positional_encoding = nn.ParameterDict()
        for i in self.variable_ids:
            self.positional_encoding[i] = nn.Parameter(
                torch.randn(
                    1,
                    positional_encoding_dim,
                    *positional_encoding_modes,
                    dtype=torch.cfloat,
                )
            )

        # each variable is represented by 1 channel + its positionnal encoding + the static field(s)
        # Note that the static field(s) are concatenated to each variable and are not considered as a separate token
        self.extended_variable_codimemsion = 1 + static_channel_dim + positional_encoding_dim

        # channel MLP to lift the input codimension to hidden_variable_codimension
        self.lifting = nn.Sequential(
            nn.Conv2d(self.extended_variable_codimemsion, hidden_lifting_channels, kernel_size=1),
            norm(hidden_lifting_channels),
            activation(),
            nn.Conv2d(hidden_lifting_channels, hidden_variable_codimension, kernel_size=1),
        )

        if per_channel_attention:
            token_dim = 1
        else:
            token_dim = hidden_variable_codimension

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                CoDABlock2D(
                    modes=modes,
                    token_dim=token_dim,
                    n_heads=n_heads,
                    activation=activation,
                    norm=norm,
                    spectral_compression_factor=spectral_compression_factor,
                )
            )

        # Permutation equivariant projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_variable_codimension, hidden_lifting_channels, kernel_size=1),
            norm(hidden_lifting_channels),
            activation(),
            nn.Conv2d(hidden_lifting_channels, 1, kernel_size=1),
        )

    def _get_positional_encoding(self, x):
        """
        Returns the positional encoding for each input variables.
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
        returns: Tensor of shape (1, num_inp_var, enc_dim, H, W, ...)
        """

        encoding_list = []
        for i in self.variable_ids:
            encoding_list.append(torch.fft.irfftn(self.positional_encoding[i], s=x.shape[-self.ndim :]))

        return torch.stack(encoding_list, dim=1)

    def _extend_variables(self, x, static_channel):
        """
        Extend the input variables by concatenating the static channel and positional encoding for each variable.
        Works only if each variable has dim=1
        Note that self._get_positional_encoding must return a tensor of shape (1, num_inp_var, enc_dim, H, W)
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W, ...)
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
        """
        x = x.unsqueeze(2)  # (batch_size, num_inp_var, 1, H, W)
        if static_channel is not None:
            # repeat the field for each variable
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]
            static_channel = static_channel.unsqueeze(1).repeat(*repeat_shape)
            x = torch.cat([x, static_channel], dim=2)  # (batch_size, num_inp_var, 1 + static_field_dim, H, W)

        # positional_encoding tensor has shape (1, num_inp_var, enc_dim, H, W)
        # Note that the econding are the same for each element in the batch
        positional_encoding = self._get_positional_encoding(x)
        repeat_shape = [1 for _ in x.shape]
        repeat_shape[0] = x.shape[0]  # repeat along the batch size to match input
        x = torch.cat([x, positional_encoding.repeat(*repeat_shape)], dim=2)
        return x

    def forward(self, x, static_channel=None):

        # concat input [B, n, H, W]
        # pos enc for each var n_enc channels (between each var)
        # add field for each channel
        # B n 1 H W + B n n_enc H W + B n n_static H W

        batch, num_inp_var, *spatial_shape = x.shape

        # concatenate  positional encodings and static fields for each variables
        # (batch_size, num_inp_var, extended_variable_codimemsion, H, W)
        x = self._extend_variables(x, static_channel)

        # Lifting each variable to higher dimensional space (hidden_variable_codimension)
        # As each variable is processed independantly, the tensor is reshaped before being sent to the the MLP
        x = x.reshape(batch * num_inp_var, self.extended_variable_codimemsion, *spatial_shape)
        x = self.lifting(x)
        x = x.reshape(batch, num_inp_var * self.hidden_variable_codimension, *spatial_shape)

        for block in self.blocks:
            x = block(x)  # batch_size, num_inp_var * extended_variable_codimemsion, H, W

        # projection to the final output ()
        x = x.reshape(batch * num_inp_var, self.hidden_variable_codimension, *spatial_shape)
        x = self.project(x)
        x = x.reshape(batch, num_inp_var, *spatial_shape)
        return x
