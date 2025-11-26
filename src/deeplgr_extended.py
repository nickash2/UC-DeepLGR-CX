import torch
import torch.nn as nn
from deeplgr import DeepLGR, SEBlock, GlobalNet


class DeepLGRExtended(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        n_external_features=21,
        n_residuals=9,
        n_filters=64,
        t_params=(12, 3, 3),
        height=32,
        width=32,
        pred_step=1,
        flag_global=True,
        predictor="td",
    ):
        super(DeepLGRExtended, self).__init__()

        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.out_channels = out_channels * pred_step
        self.flag_global = flag_global
        self.predictor = predictor
        self.n_external_features = n_external_features

        flow_channels = sum(t_params) * in_channels

        self.conv1 = nn.Conv2d(flow_channels, n_filters, 3, 1, 1)

        self.external_embedding = nn.Sequential(
            nn.Linear(n_external_features, 10),
            nn.ReLU(),
            nn.Linear(10, self.out_channels * height * width),
        )

        se_blocks = []
        for _ in range(n_residuals):
            se_blocks.append(SEBlock(n_filters))
        self.senet = nn.Sequential(*se_blocks)

        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)

        if flag_global:
            self.glonet = GlobalNet(64, 64, (1, 2, 4, 8), height, width)

        if predictor == "td":
            d1 = 16
            d2 = 16
            d3 = 32
            self.core = nn.Parameter(torch.FloatTensor(d1, d2, d3))
            self.F = nn.Parameter(torch.FloatTensor(d3, n_filters * self.out_channels))
            self.H = nn.Parameter(torch.FloatTensor(d1, height))
            self.W = nn.Parameter(torch.FloatTensor(d2, width))
            nn.init.normal_(self.core, 0, 0.02)
            nn.init.normal_(self.F, 0, 0.02)
            nn.init.normal_(self.H, 0, 0.02)
            nn.init.normal_(self.W, 0, 0.02)
        elif predictor == "md":
            self.L = nn.Parameter(torch.FloatTensor(height * width, 10))
            self.R = nn.Parameter(torch.FloatTensor(10, n_filters * self.out_channels))
            nn.init.normal_(self.L, 0, 0.02)
            nn.init.normal_(self.R, 0, 0.02)
        else:
            self.output_conv = nn.Sequential(
                nn.Conv2d(n_filters, self.out_channels, 1, 1, 0)
            )

    def forward(self, inputs, external_features):
        out = torch.cat(inputs, dim=1)
        b = out.shape[0]

        out1 = self.conv1(out)
        out = self.senet(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.flag_global:
            out = self.glonet(out)

        if self.predictor == "td":
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1)
            region_param = torch.matmul(self.core, self.F)
            region_param = region_param.permute(1, 2, 0)
            region_param = torch.matmul(region_param, self.H)
            region_param = region_param.permute(1, 2, 0)
            region_param = torch.matmul(region_param, self.W)
            region_param = region_param.unsqueeze(0).repeat(b, 1, 1, 1)
            region_param = region_param.reshape(
                b, -1, self.n_filters, self.height * self.width
            ).permute(0, 3, 2, 1)
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels)
            out = torch.sum(region_features * region_param, 2).reshape(
                b, self.height, self.width, -1
            )
            out = out.permute(0, 3, 1, 2)
        elif self.predictor == "md":
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1)
            region_param = torch.matmul(self.L, self.R).unsqueeze(0)
            region_param = region_param.repeat(b, 1, 1).reshape(
                b, -1, self.n_filters, self.out_channels
            )
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels)
            out = torch.sum(region_features * region_param, 2).reshape(
                b, self.height, self.width, -1
            )
            out = out.permute(0, 3, 1, 2)
        else:
            out = self.output_conv(out)

        external_out = self.external_embedding(external_features)
        external_out = external_out.reshape(
            b, self.out_channels, self.height, self.width
        )

        out = out + external_out

        return out


def create_baseline_model(t_params=(12, 3, 3), height=32, width=32):
    model = DeepLGR(
        in_channels=2,
        out_channels=2,
        n_residuals=9,
        n_filters=64,
        t_params=t_params,
        height=height,
        width=width,
        pred_step=1,
        flag_global=True,
        predictor="td",
    )
    return model


def create_extended_model(
    t_params=(12, 3, 3), height=32, width=32, n_external_features=21
):
    model = DeepLGRExtended(
        in_channels=2,
        out_channels=2,
        n_external_features=n_external_features,
        n_residuals=9,
        n_filters=64,
        t_params=t_params,
        height=height,
        width=width,
        pred_step=1,
        flag_global=True,
        predictor="td",
    )
    return model
