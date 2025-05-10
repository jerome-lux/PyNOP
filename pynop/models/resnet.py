import torch.nn as nn

from pynop.core.ops import ConvLayer
from pynop.core.blocks import ResBlock, BottleneckBlock
from pynop.core.norm import LayerNorm2d
from pynop.core.utils import make_tuple
from pynop.core.activations import build_activation


class Resnet(nn.Module):
    """
    Resnet with classic conv redidual block
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
    ):

        use_bias = make_tuple(use_bias, 2)
        norm = make_tuple(norm, 2)
        activation = make_tuple(activation, 2)

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


class Resnetb(nn.Module):
    """
    Resnet with bottleneck redidual block
    can be used to make a resnext when groups is > 1
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
        stem_norm="bn",
        stem_act="silu",
        preact=False,
        se_module=False,
        bottleneck_ratio=4,
        activation="silu",
        use_bias=False,
        norm="bn",
        norm_kwargs={},
        fc_layers=[],
        groups=1,
        build_head=False,
        head_activation=None,
        dropout=0.5,
        downsampling_method="maxpooling",
    ):
        # parameter of each conv layer in the bottleneck block
        use_bias = make_tuple(use_bias, 3)
        norm = make_tuple(norm, 3)
        activation = make_tuple(activation, 3)

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        if not preact:
            self.stem = ConvLayer(
                in_channels,
                stem_filters,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                norm=stem_norm,
                activation=stem_act,
                norm_kwargs=norm_kwargs,
                use_bias=False,
            )
        else:
            self.stem = ConvLayer(
                in_channels,
                stem_filters,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                norm=None,
                activation=None,
                norm_kwargs=norm_kwargs,
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
                self.ops.append(
                    BottleneckBlock(
                        input_channels,
                        dims[i],
                        stride=stride,
                        bottleneck_ratio=bottleneck_ratio,
                        se=se_module,
                        norm=norm,
                        use_bias=use_bias,
                        activation=activation,
                        norm_kwargs=norm_kwargs,
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

    return Resnetb(dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], **kwargs)


def resnext50(**kwargs):

    return Resnetb(dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], groups=32, bottleneck_ratio=2, **kwargs)
