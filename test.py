import torch
import torchaudio
from config import set_params
from wavenet.model import build_wavenet
from wavenet.utils import MuLawQuantization


def main():
    # set params
    params = set_params()
    params.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load spectrogram
    melspec = torch.load(params.example_spectrogram).to(params.device)

    # load model checkpoint
    model = build_wavenet(params).to(params.device)
    checkpoint = torch.load(params.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # generate waveform
    quantizer = MuLawQuantization(params.mu)
    mu_law = model.inference(melspec, quantizer)
    waveform = quantizer.inverse(mu_law)
    torchaudio.save(params.example_audio, waveform, params.sample_rate)


if __name__ == '__main__':
    main()
