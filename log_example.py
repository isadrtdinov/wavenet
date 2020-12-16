import os
import wandb
import torch
import torchaudio
from config import set_params


def main():
    params = set_params()
    os.environ['WANDB_API_KEY'] = '871c3c897b41570255fd1829026512c6f84d7a7f'
    wandb.init(project=params.wandb_project)

    ground_truth, _ = torchaudio.load(params.music_audio)
    generated, _ = torchaudio.load(params.generated_audio)

    ground_truth = ground_truth.squeeze(0).numpy()
    generated = generated.squeeze(0).numpy()

    wandb.log({'ground truth audio': wandb.Audio(ground_truth, sample_rate=params.sample_rate),
               'generated audio': wandb.Audio(generated, sample_rate=params.sample_rate)})


if __name__ == '__main__':
    main()
