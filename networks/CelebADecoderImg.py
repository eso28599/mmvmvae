import torch.nn as nn

class ResidualBlock2dTransposeConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernelsize,
        stride,
        padding,
        dilation,
        o_padding,
        upsample,
        a=1.0,
        b=1.0,
    ):
        super(ResidualBlock2dTransposeConv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            channels_in,
            channels_in,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=False,
        )
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            output_padding=o_padding,
        )
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.upsample = upsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out
        return out


def res_block_gen(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    o_padding,
    dilation,
    a_val=1.0,
    b_val=1.0,
):
    upsample = None
    # upsampling is always performed in the decoder for celeba, since stride=2 for all res blocks in the decoder and in_channels!=out_channels for all res blocks in the decoder
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=o_padding,
            ),
            nn.BatchNorm2d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock2dTransposeConv(
            in_channels,
            out_channels,
            kernelsize=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            o_padding=o_padding,
            upsample=upsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class DataGeneratorImg(nn.Module):
    def __init__(self, cfg, a, b):
        super(DataGeneratorImg, self).__init__()
        self.resblock1 = res_block_gen(
            5 * cfg.dataset.filter_dim_img,
            4 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=1,
            padding=0,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock2 = res_block_gen(
            4 * cfg.dataset.filter_dim_img,
            3 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock3 = res_block_gen(
            3 * cfg.dataset.filter_dim_img,
            2 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock4 = res_block_gen(
            2 * cfg.dataset.filter_dim_img,
            1 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.conv = nn.ConvTranspose2d(
            cfg.dataset.filter_dim_img,
            cfg.dataset.image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )

    def forward(self, feats):
        d = self.resblock1(feats)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        d = self.conv(d)
        return d
