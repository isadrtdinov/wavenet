import torch
from wavenet.model import build_wavenet
from wavenet.train import Trainer
from wavenet.utils import (
    set_random_seed,
    load_data,
    split_data,
    LJSpeechDataset
)
from config import set_params


def main():
    # set params and random seed
    params = set_params()
    set_random_seed(params.random_seed)
    params.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params.verbose:
        print('Using device', params.device)

    # load and split data
    data = load_data(params.metadata_file)
    train_data, valid_data = split_data(data, params.valid_ratio)
    if params.verbose:
        print('Data loaded and split')

    # prepare dataloaders
    train_dataset = LJSpeechDataset(labels=train_data, params=params)
    valid_dataset = LJSpeechDataset(labels=valid_data, params=params)

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
