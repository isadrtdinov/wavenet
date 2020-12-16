import torch
from wavenet.model import build_wavenet
from wavenet.train import Trainer
from wavenet.utils import (
    set_random_seed,
    load_data,
    split_data,
    ElectroDataset
)
from config import set_params


def main():
    # set params and random seed
    params = set_params()
    set_random_seed(params.random_seed)
    params.device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    if params.verbose:
        print('Using device', params.device)

    # prepare dataloaders
    train_dataset = ElectroDataset(params=params, mode='train')
    valid_dataset = ElectroDataset(params=params, mode='valid')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, pin_memory=True)
    if params.verbose:
        print('Data loaders prepared')

    model = build_wavenet(params)
    trainer = Trainer(model, params)
    if params.load_model:
        trainer.load_checkpoint(params.model_checkpoint)

    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
