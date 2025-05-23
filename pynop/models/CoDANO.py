from typing import Callable, Union, Sequence
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from pynop.core.blocks import CoDABlock2D
from pynop.core.ops import CartesianEmbedding


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
        fixed_pos_encoding: bool = True,
        positional_encoding_dim: int = 8,
        positional_encoding_modes: Sequence = (16, 16),
        static_channel_dim: int = 0,
        modes: tuple[int, int] = (16, 16),
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
        self.fixed_pos_encoding = fixed_pos_encoding

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
            input tensor of shape (batch_size, num_inp_var, H, W
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
        Returns
        -------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, C, H, W)
            where C is the number of channels after concatenation (1 + grid_pos_ch + enc_dim + static_channel_dim)
        """

        if self.fixed_pos_encoding:
            height, width = x.shape[-2:]
            batch_size = x.shape[0]
            # Generate 1D coordinate vectors normalized to [-1, 1]
            x_lin = torch.linspace(-1, 1, steps=width, device=x.device)  # Shape (W,)
            y_lin = torch.linspace(-1, 1, steps=height, device=x.device)  # Shape (H,)
            grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")
            grid_encoding = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)
            grid_encoding = grid_encoding.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 2, H, W)
            grid_encoding = grid_encoding.expand(batch_size, -1, -1, -1)  # Shape (batch_size, 2, H, W)
            grid_encoding.unsqueeze(1)  # (batch_size, 1, 2, H, W)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]  # repeat along the variable axis to match input
            grid_encoding = grid_encoding.repeat(*repeat_shape)  # (batch_size, num_inp_var, 2, H, W)
            x = x.unsqueeze(2)  # (batch_size, num_inp_var, 1, H, W)
            x = torch.cat([x, grid_encoding], dim=2)  # (batch_size, num_inp_var, 1 + 2, H, W)
        else:
            x = x.unsqueeze(2)

        # positional_encoding tensor has shape (1, num_inp_var, enc_dim, H, W)
        # Note that the encodings are the same for each element in the batch
        positional_encoding = self._get_positional_encoding(x)
        repeat_shape = [1 for _ in x.shape]
        repeat_shape[0] = x.shape[0]  # repeat along the batch size to match input
        x = torch.cat(
            [x, positional_encoding.repeat(*repeat_shape)], dim=2
        )  # (batch_size, num_inp_var, 1 + static_field_dim + 2 + pos_encoding_ch, H, W)

        if static_channel is not None:
            # repeat the field for each variable
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]
            static_channel = static_channel.unsqueeze(1).repeat(
                *repeat_shape
            )  # (batch_size, num_inp_var, static_channel_dim, H, W)
            x = torch.cat([x, static_channel], dim=2)

        return x

    def forward(self, x, static_channel=None, return_coords=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W)
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W)
        return_coords : bool
            if True, the model outputs the grid encoding to allow  for the computation of the physical loss
        Returns
        -------
        x : torch.Tensor
            output tensor of shape (batch_size, num_inp_var, H, W)
        """

        # concat input [B, n, H, W]
        # pos enc for each var n_enc channels (between each var)
        # add field for each channel
        # B n 1 H W + B n n_enc H W + B n n_static H W

        batch, num_inp_var, *spatial_shape = x.shape

        # concatenate  positional encodings and static fields for each variables
        # (batch_size, num_inp_var, extended_variable_codimemsion, H, W)
        x = self._extend_variables(x, static_channel)
        if return_coords and self.fixed_pos_encoding:
            coords = x[:, :, 1:2, :, :]  # (batch_size, num_inp_var, 2, H, W)
        elif return_coords and not self.fixed_pos_encoding:
            raise ValueError(
                "return_coords is set to True, but fixed_pos_encoding is set to False. "
                "This means that the model does not have a grid encoding to return."
            )

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

        if return_coords and self.fixed_pos_encoding:
            return x, coords
        else:
            return x
