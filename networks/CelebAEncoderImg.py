import torch.nn as nn

class ResidualBlock2dConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernelsize,
        stride,
        padding,
        dilation,
        downsample,
        a=1.0,
        b=1.0,
    ):
        super(ResidualBlock2dConv, self).__init__()
        self.conv1 = nn.Conv2d(
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
        self.conv2 = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.downsample = downsample
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.a * residual + self.b * out
        return out


def make_res_block_feature_extractor(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    dilation,
    a_val=1.0, # fixed as 1 throughout exps
    b_val=1.0, # fixed as 1 throughout exps
):
    downsample = None
    # in_channels!=out_channels for all exps, so downsampling is always performed
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock2dConv(
            in_channels,
            out_channels,
            kernelsize,
            stride,
            padding,
            dilation,
            downsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
  # filter_dim_img is fixed as 64 throughout exps
  # num_layers_img is fixed as 5 throughout exps
  
    def __init__(self, cfg, a, b):
        super(FeatureExtractorImg, self).__init__()
        self.conv1 = nn.Conv2d(
            cfg.dataset.image_channels,
            cfg.dataset.filter_dim_img,
            kernel_size=3,
            stride=2,
            padding=2,
            dilation=1,
            bias=False,
        )
        self.resblock1 = make_res_block_feature_extractor(
            cfg.dataset.filter_dim_img,
            2 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock2 = make_res_block_feature_extractor(
            2 * cfg.dataset.filter_dim_img,
            3 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock3 = make_res_block_feature_extractor(
            3 * cfg.dataset.filter_dim_img,
            4 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock4 = make_res_block_feature_extractor(
            4 * cfg.dataset.filter_dim_img,
            5 * cfg.dataset.filter_dim_img,
            kernelsize=4,
            stride=2,
            padding=0,
            dilation=1,
            a_val=a,
            b_val=b,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        return out
