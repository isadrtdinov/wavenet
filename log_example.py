import wandb
import torch
import torchaudio
from config import set_params


def main():
    params = set_params()
    wandb.init(project=params.wandb_project)

    ground_truth = torchaudio.load(params.ground_truth_audio).squeeze(0).numpy()
    generated = torchaudio.load(params.example_audio).squeeze(0).numpy()
    spectrogram = torch.load(params.example_spectrogram).squeeze(0).numpy()

    wandb.log({'ground truth audio': wandb.Audio(ground_truth),
               'generated audio': wandb.Audio(generated),
               'spectrogram': wandb.Image(spectrogram)})


if __name__ == '__main__':
    main()
