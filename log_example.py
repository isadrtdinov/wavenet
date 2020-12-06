import wandb
import torch
import torchaudio
from config import set_params


def main():
    params = set_params()
    wandb.init(project=params.wandb_project)

    ground_truth, _ = torchaudio.load(params.ground_truth_audio)
    generated, _ = torchaudio.load(params.example_audio)
    spectrogram = torch.load(params.example_spectrogram)

    ground_truth = ground_truth.squeeze(0).numpy()
    generated = generated.squeeze(0).numpy()
    spectrogram = spectrogram.squeeze(0).numpy()

    wandb.log({'ground truth audio': wandb.Audio(ground_truth, sample_rate=params.sample_rate),
               'generated audio': wandb.Audio(generated, sample_rate=params.sample_rate),
               'spectrogram': wandb.Image(spectrogram)})


if __name__ == '__main__':
    main()
