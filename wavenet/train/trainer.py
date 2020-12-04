import os
import wandb
import torch
from torch import nn
from ..utils.quantization import MuLawQuantization
from ..utils.spectrogram import MelSpectrogram


class Trainer(object):
    def __init__(self, model, params):
        self.params = params
        self.model = model.to(params.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.lr,
                                          weight_decay=params.weight_decay)
        self.spectrogramer = MelSpectrogram(params).to(params.device)
        self.quantizer = MuLawQuantization(params.mu).to(params.device)
        self.criterion = nn.CrossEntropyLoss()

        if params.use_wandb:
            wandb.init(project=params.wandb_project)
            wandb.watch(self.model)

    def save_checkpoint(self, epoch):
        if not os.path.isdir(self.params.checkpoint_dir):
            os.mkdir(self.params.checkpoint_dir)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }, self.params.checkpoint_template.format(epoch))

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])

    def log_metrics(self, train_metrics, valid_metrics):
        wandb.log({'train loss': train_metrics[0], 'train accuracy': train_metrics[1],
                   'valid loss': valid_metrics[0], 'valid accuracy': valid_metrics[1]})

    def process_epoch(self, loader, train=True):
        running_loss, running_accuracy = 0.0, 0.0

        for waveforms in loader:
            with torch.no_grad():
                waveforms = waveforms.to(self.params.device)
                melspecs = self.spectrogramer(waveforms)

                mu_law = self.quantizer(waveforms)
                zeros = torch.zeros((mu_law.shape[0], 1)).to(self.params.device)
                inputs = torch.cat([zeros, waveforms[:, :-1]], dim=1)
                targets = self.quantizer.quantize(mu_law[:, 1:])

            with torch.set_grad_enabled(train):
                logits = self.model(inputs, melspecs)
                targets = targets[:, :logits.shape[-1]]
                loss = self.criterion(logits, targets)
                accuracy = (torch.argmax(logits, dim=1) == targets).to(torch.float).mean()

            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * waveforms.shape[0]
            running_accuracy += accuracy.item() * waveforms.shape[0]

        running_loss /= len(loader.dataset)
        running_accuracy /= len(loader.dataset)
        return running_loss, running_accuracy

    def train(self, train_loader, valid_loader):
        for epoch in range(self.params.start_epoch, self.params.start_epoch + self.params.num_epochs):
            train_metrics = self.process_epoch(train_loader, train=True)
            valid_metrics = self.process_epoch(valid_loader, train=False)

            if self.params.use_wandb:
                self.log_metrics(train_metrics, valid_metrics)
