import torch
import torchaudio
from config import set_params
from wavenet.model import build_wavenet
from wavenet.utils import MuLawQuantization


def main():
    # set params
    params = set_params()
    params.device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")

    # load model checkpoint
    model = build_wavenet(params).to(params.device)
    checkpoint = torch.load(params.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # generate waveform
    waveform = torchaudio.load(params.music_audio)
    quantizer = MuLawQuantization(params.mu).to(params.device)
    mu_law = model.inference(waveform, quantizer, params.generation_length)
    waveform = quantizer.inverse(mu_law).cpu()
    torchaudio.save(params.generated_audio, waveform, params.sample_rate)


if __name__ == '__main__':
    main()
