import torch
from torch import nn
from ..utils.utils import init_xavier


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, dilation=dilation)
        init_xavier(self.conv)

    def forward(self, inputs):
        padding = torch.zeros((inputs.shape[0], inputs.shape[1], self.padding)).to(inputs.device)
        inputs = torch.cat([padding, inputs], dim=-1)
        return self.conv(inputs)


class WaveNetBlock(nn.Module):
    def __init__(self, causal_kernel=2, causal_dilation=1,
                 residual_channels=120, skip_channels=240):
        super().__init__()
        self.residual_channels = residual_channels
        self.causal_conv = CausalConv1d(residual_channels, 2 * residual_channels,
                                        causal_kernel, causal_dilation)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

        init_xavier(self.skip_conv)
        init_xavier(self.residual_conv)

    def forward(self, inputs):
        # inputs: (batch_size, residual_channels, audio_length)

        conv_inputs = self.causal_conv(inputs)
        # conv_inputs: (batch_size, 2 * residual_channels, audio_length)

        gates = torch.tanh(conv_inputs[:, :self.residual_channels]) * \
                torch.sigmoid(conv_inputs[:, self.residual_channels:])
        # gates: (batch_size, 2 * residual_channels, audio_length)

        skips = self.skip_conv(gates)
        residuals = self.residual_conv(gates) + inputs
        # skips: (batch_size, skip_channels, audio_length)
        # residuals: (batch_size, residual_channels, audio_length)

        return skips, residuals
