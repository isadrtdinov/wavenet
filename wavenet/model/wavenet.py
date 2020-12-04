import torch
from torch import nn
from .layers import WaveNetBlock


class WaveNet(nn.Module):
    def __init__(self, num_mels=80, num_quants=256, win_length=1024, hop_length=256,
                 upsample_kernel=800, residual_channels=120, skip_channels=240,
                 causal_kernel=2, num_blocks=16, dilation_cycle=8):
        super().__init__()
        assert num_blocks % dilation_cycle == 0

        padding = (upsample_kernel + 4 * hop_length - win_length)
        self.upsample = nn.ConvTranspose1d(in_channels=num_mels, out_channels=num_mels,
                                           kernel_size=upsample_kernel, stride=hop_length,
                                           padding=padding)
        self.input_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        num_cycles = num_blocks // dilation_cycle
        dilations = [2 ** power for power in range(dilation_cycle)]
        self.blocks = []
        for _ in range(num_cycles):
            self.blocks += [WaveNetBlock(num_mels, causal_kernel, dilation,
                                         residual_channels, skip_channels)
                            for dilation in dilations]

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, num_quants, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_quants, num_quants, kernel_size=1),
        )

    def forward(self, waveforms, melspecs):
        # waveforms: (batch_size, audio_length)
        # melspecs: (batch_size, num_mels, frames_length)

        conditions = self.upsample(melspecs)
        length = min(waveforms.shape[-1], conditions.shape[-1])
        waveforms = waveforms[..., :length]
        conditions = conditions[..., :length]
        # waveforms: (batch_size, audio_length)
        # conditions: (batch_size, num_mels, audio_length)

        skips_list = []
        residuals = self.input_conv(waveforms.unsqueeze(1))
        for block in self.blocks:
            skips, residuals = block(residuals, conditions)
            skips_list += [skips]
            # skips: (batch_size, skip_channels, audio_length)
            # residuals: (batch_size, residual_channels, audio_length)

        skips = torch.stack(skips_list, dim=0).sum(dim=0)
        logits = self.head(skips)
        # logits: (batch_size, num_quants, audio_length)

        return logits

    def inference(self, melspecs):
        # melspecs: (batch_size, num_mels, frames_length)
        pass


def build_wavenet(params):
    return WaveNet(params.num_mels, params.mu, params.win_length, params.hop_length,
                   params.upsample_kernel, params.residual_channels, params.skip_channels,
                   params.causal_kernel, params.num_blocks, params.dilation_cycle)
