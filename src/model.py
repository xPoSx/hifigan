import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F
import src.config as config


class ResBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dilations)):
            inner_conv = []
            for j in range(len(dilations[i])):
                inner_conv.append(nn.LeakyReLU(0.1))
                inner_conv.append(nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilations[i][j],
                                            padding=(dilations[i][j] * (kernel_size - 1)) // 2))
            self.layers.append(nn.Sequential(*inner_conv))

    def forward(self, input):
        out = input
        for i in range(len(self.layers)):
            out = out + self.layers[i](out)
        return out


class MRF(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([ResBlock(hidden_size, 3, config.gen_dilations[0]),
                                     ResBlock(hidden_size, 5, config.gen_dilations[1]),
                                     ResBlock(hidden_size, 7, config.gen_dilations[2])])

    def forward(self, input):
        out = self.layers[0](input)
        for i in range(2):
            out = out + self.layers[i + 1](input)
        out /= 3
        return out


class Generator(nn.Module):
    def __init__(self, in_ch, kernel_sizes=config.gen_kernel_sizes):
        super().__init__()
        self.input_conv = nn.Conv1d(in_ch, 256, 7, padding=3)

        self.layers = nn.Sequential(*[nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(256 // (2 ** i), 256 // (2 ** (i + 1)), kernel_sizes[i], kernel_sizes[i] // 2,
                               kernel_sizes[i] // 4),
            MRF(256 // (2 ** (i + 1)))) for i in range(3)])

        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.input_conv(input)
        out = self.layers(out)
        out = self.out_conv(out)
        return out


class SubPD(nn.Module):
    def __init__(self, period, kernel_size=config.subpd_kernel_size, stride=config.subpd_stride, padding=config.subpd_padding):
        super().__init__()
        self.p = period
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(nn.Sequential(
                weight_norm(nn.Conv2d(config.subpd_channels[i], config.subpd_channels[i + 1], kernel_size, stride, padding)),
                nn.LeakyReLU(0.1)))
        self.layers.append(nn.Sequential(
            weight_norm(nn.Conv2d(config.subpd_channels[-1], config.subpd_channels[-1], kernel_size, 1, padding)),
            nn.LeakyReLU(0.1)))
        self.layers.append(weight_norm(nn.Conv2d(1024, 1, (3, 1))))

    def forward(self, input):
        pad = self.p - (input.shape[-1] % self.p)
        out = F.pad(input, (0, pad), "reflect")
        out = out.view(input.shape[0], input.shape[1], -1, self.p)

        features = []
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            features.append(out)
        return torch.flatten(out, 1, -1), features


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.ModuleList([
            SubPD(2),
            SubPD(3),
            SubPD(5),
            SubPD(7),
            SubPD(11)
        ])

    def forward(self, true, pred):
        true_outs = []
        true_features = []
        pred_outs = []
        pred_features = []
        for i in range(len(self.disc)):
            true_out, true_feature = self.disc[i](true)
            pred_out, pred_feature = self.disc[i](pred)
            true_outs.append(true_out)
            true_features.append(true_feature)
            pred_outs.append(pred_out)
            pred_features.append(pred_feature)
        return true_outs, true_features, pred_outs, pred_features


class SubSD(nn.Module):
    def __init__(self, spectral=False):
        super().__init__()
        if spectral:
            norm = spectral_norm
        else:
            norm = weight_norm
        self.layers = nn.ModuleList()
        for i in range(6):
            self.layers.append(
                nn.Sequential(
                    norm(nn.Conv1d(
                        config.subsd_channels[i], config.subsd_channels[i + 1], config.subsd_kernels[i],
                        config.subsd_strides[i], padding=config.subsd_kernels[i] // 2, groups=config.subsd_groups[i]
                    )),
                    nn.LeakyReLU(0.1)
                )
            )
        self.layers.append(nn.Sequential(
                    norm(nn.Conv1d(
                        config.subsd_channels[-1], config.subsd_channels[-1], config.subsd_kernels[-1],
                        config.subsd_strides[-1], padding=config.subsd_kernels[-1] // 2, groups=config.subsd_groups[-1]
                    )),
                    nn.LeakyReLU(0.1)
                ))
        self.layers.append(norm(nn.Conv1d(config.subsd_channels[-1], 1, 3, padding=1)))

    def forward(self, input):
        out = input
        features = []
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            features.append(out)
        return torch.flatten(out, 1, -1), features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.ModuleList([
            SubSD(True)
        ])
        for i in range(2):
            self.disc.append(
                nn.Sequential(
                    nn.AvgPool1d(4, 2, 2),
                    SubSD()
                )
            )

    def forward(self, true, pred):
        true_outs = []
        true_features = []
        pred_outs = []
        pred_features = []
        for i in range(len(self.disc)):
            true_out, true_feature = self.disc[i](true)
            pred_out, pred_feature = self.disc[i](pred)
            true_outs.append(true_out)
            true_features.append(true_feature)
            pred_outs.append(pred_out)
            pred_features.append(pred_feature)
        return true_outs, true_features, pred_outs, pred_features
