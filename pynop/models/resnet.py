import torch.nn as nn

from pynop.core.ops import ConvLayer
from pynop.core.blocks import ResBlock, BottleneckBlock
from pynop.core.activations import build_activation


class Resnet(nn.Module):
    """
    Resnet backbone built from either basic or bottleneck residual blocks.

    Select the building block with ``block_type``:

    - ``"basic"`` (default) — :class:`~pynop.core.blocks.ResBlock`
      (two 3x3 convolutions, classic ResNet-18/34 style).
    - ``"bottleneck"`` — :class:`~pynop.core.blocks.BottleneckBlock`
      (1x1 -> 3x3 -> 1x1 with an optional Squeeze-and-Excitation gate,
      ResNet-50 style). The mid-channel width is reduced by
      ``bottleneck_ratio`` and the SE gate is toggled with ``se``.

    Each stage's first block downsamples (``stride=2``). The shortcut
    downsampler is selected by ``downsampling_method``, one of
    ``"maxpooling"`` (default) or ``"stridedconv"`` (see
    :data:`pynop.core.blocks.DOWNSAMPLING_LAYERS`).

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes (used by the head).
        depths: Number of blocks in each stage.
        dims: Number of output channels in each stage.
        stem_filters: Number of channels in the stem (first) convolution.
        stem_kernel_size: Stem kernel size.
        stem_stride: Stem stride.
        stem_act: Stem activation class.
        stem_norm: Stem normalization class.
        activation: Activation class for the residual blocks.
        use_bias: Conv bias flag for the residual blocks.
        norm: Normalization class for the residual blocks.
        fc_layers: Fully-connected hidden widths of the head (``[]`` = none).
        head_activation: Activation class between head linear layers.
        groups: Convolution groups (``> 1`` gives a ResNeXt-style block).
        build_head: If ``True``, the forward pass also runs the classification head.
        dropout: Dropout rate applied inside the head.
        downsampling_method: Shortcut downsampler for the stage blocks (see above).
        block_type: ``"basic"`` or ``"bottleneck"`` (see above).
        bottleneck_ratio: Mid-channel reduction of the bottleneck block
            (only used when ``block_type="bottleneck"``).
        se: Insert a Squeeze-and-Excitation gate in the bottleneck block
            (only used when ``block_type="bottleneck"``; default ``False`` for a
            standard ResNet).

    Raises:
        ValueError: If ``block_type`` is not one of ``"basic"`` / ``"bottleneck"``.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_filters=64,
        stem_kernel_size=7,
        stem_stride=2,
        stem_act=nn.SiLU,
        stem_norm=nn.BatchNorm2d,
        activation=nn.SiLU,
        use_bias=False,
        norm=nn.BatchNorm2d,
        fc_layers=[],
        head_activation=None,
        groups=1,
        build_head=False,
        dropout=0.5,
        downsampling_method="maxpooling",
        block_type="basic",
        bottleneck_ratio=4,
        se=False,
    ):

        if block_type not in ("basic", "bottleneck"):
            raise ValueError(
                f"block_type must be 'basic' or 'bottleneck', got {block_type!r}."
            )

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.stem = ConvLayer(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=stem_norm,
            activation=stem_act,
            use_bias=False,
        )

        self.ops = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                if block_type == "basic":
                    self.ops.append(
                        ResBlock(
                            input_channels,
                            dims[i],
                            nconv=2,
                            stride=stride,
                            norm=norm,
                            use_bias=use_bias,
                            activation=activation,
                            groups=groups,
                            downsampling_method=downsampling_method,
                        )
                    )
                else:  # bottleneck
                    self.ops.append(
                        BottleneckBlock(
                            input_channels,
                            dims[i],
                            bottleneck_ratio=bottleneck_ratio,
                            stride=stride,
                            norm=norm,
                            use_bias=use_bias,
                            activation=activation,
                            se=se,
                            groups=groups,
                            downsampling_method=downsampling_method,
                        )
                    )
        self.head = nn.ModuleList()
        if fc_layers is not None:
            for depth in fc_layers:
                self.head.append(nn.Linear(dims[-1], depth, bias=True))
                head_act_layer = build_activation(head_activation)
                if head_act_layer is not None:
                    self.head.append(head_activation)
                if dropout > 0:
                    self.head.append(nn.Dropout(dropout))
                dims[-1] = depth

        self.head.append(nn.Linear(dims[-1], num_classes))

    def forward_features(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x)
        return x

    def forward_head(self, x):
        for op in self.head:
            x = op(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.forward_head(x)
        return x


def resnet18(**kwargs):

    return Resnet(dims=[64, 128, 256, 512], depths=[2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):

    return Resnet(dims=[64, 128, 256, 512], depths=[3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):

    return Resnet(block_type="bottleneck", dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], **kwargs)


def resnext50(**kwargs):

    return Resnet(block_type="bottleneck", groups=32, bottleneck_ratio=2, dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], **kwargs)
