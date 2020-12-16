import torch
from torch import nn
from .layers import WaveNetBlock
from ..utils.utils import init_xavier


class WaveNet(nn.Module):
    def __init__(self, num_quants=256, residual_channels=120, skip_channels=240,
                 causal_kernel=2, num_blocks=16, dilation_cycle=8):
        super().__init__()
        assert num_blocks % dilation_cycle == 0

        self.input_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        num_cycles = num_blocks // dilation_cycle
        dilations = [2 ** power for power in range(dilation_cycle)]
        self.blocks = []
        for _ in range(num_cycles):
            self.blocks += [WaveNetBlock(causal_kernel, dilation,
                                         residual_channels, skip_channels)
                            for dilation in dilations]
        self.blocks = nn.ModuleList(self.blocks)
        self.receptive_field = (2 ** dilation_cycle - 1) * num_cycles + 1

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, num_quants, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_quants, num_quants, kernel_size=1),
        )

        init_xavier(self.input_conv)
        init_xavier(self.head[1])
        init_xavier(self.head[3])

    def forward(self, waveforms):
        # waveforms: (batch_size, audio_length)

        skips_list = []
        residuals = self.input_conv(waveforms.unsqueeze(1))
        for block in self.blocks:
            skips, residuals = block(residuals)
            skips_list += [skips]
            # skips: (batch_size, skip_channels, audio_length)
            # residuals: (batch_size, residual_channels, audio_length)

        skips = torch.stack(skips_list, dim=0).sum(dim=0)
        logits = self.head(skips)
        # logits: (batch_size, num_quants, audio_length)

        return logits

    def inference(self, waveforms, quantizer, length):
        # waveforms: (batch_size, audio_length)

        for i in range(length):
            if i % 100 == 0:
                print(i, '/', length, sep='')
            input_waveforms = waveforms[:, -self.receptive_field:]
            logits = self.forward(input_waveforms)
            # logits: ((batch_size, num_quants, receptive_field)

            quants = torch.argmax(logits[..., -1].detach(), dim=1).unsqueeze(-1)
            waveforms = torch.cat([waveforms, quantizer.dequantize(quants)], dim=1)
        # waveforms: (batch_size, audio_length)

        return waveforms


def build_wavenet(params):
    return WaveNet(params.mu, params.residual_channels, params.skip_channels,
                   params.causal_kernel, params.num_blocks, params.dilation_cycle)
