import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                              dilation=dilation, padding=self.padding)

    def forward(self, inputs):
        return self.conv(inputs)[..., :-self.padding]


class WaveNetBlock(nn.Module):
    def __init__(self, num_mels=80, causal_kernel=2, causal_dilation=1,
                 residual_channels=120, skip_channels=240):
        super().__init__()
        self.causal_conv = CausalConv1d(residual_channels, 2 * residual_channels,
                                        causal_kernel, causal_dilation)
        self.conditional_conv = nn.Conv1d(num_mels, 2 * residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(2 * residual_channels, skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(2 * residual_channels, residual_channels, kernel_size=1)

    def forward(self, inputs, conditions):
        # inputs: (batch_size, residual_channels, audio_length)
        # conditions: (batch_size, num_mels, audio_length)

        conv_inputs = self.causal_conv(inputs)
        conv_conditions = self.conditional_conv(conditions)
        # conv_inputs, conv_conditions: (batch_size, 2 * residual_channels, audio_length)

        gates = conv_inputs + conv_conditions
        gates = torch.tanh(gates) * torch.sigmoid(gates)
        # gates: (batch_size, 2 * residual_channels, audio_length)

        skips = self.skip_conv(gates)
        residuals = self.residual_conv(gates) + inputs
        # skips: (batch_size, skip_channels, audio_length)
        # residuals: (batch_size, residual_channels, audio_length)

        return skips, residuals
