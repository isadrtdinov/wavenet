import torch
from torch import nn
import torchaudio
import librosa


class MelSpectrogram(nn.Module):
    def __init__(self, params):
        super(MelSpectrogram, self).__init__()

        self.pad_value = params.pad_value

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=params.sample_rate,
            win_length=params.win_length,
            hop_length=params.hop_length,
            n_fft=params.n_fft,
            f_min=params.f_min,
            f_max=params.f_max,
            n_mels=params.num_mels
        )

        # There is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = params.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=params.sample_rate,
            n_fft=params.n_fft,
            n_mels=params.num_mels,
            fmin=params.f_min,
            fmax=params.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio):
        # audio: (batch_size, length)

        melspec = self.mel_spectrogram(audio).clamp_(min=self.pad_value).log_()
        # melspec: (batch_size, num_mels, frames_length)

        return melspec


def get_spectrogram_lengths(audio_lengths, params):
    return (audio_lengths - params.win_length) // params.hop_length + 5

