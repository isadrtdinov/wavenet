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
        self.train_step = 0
        self.train_loss, self.train_accuracy = 0.0, 0.0
        self.valid_loss, self.valid_accuracy = 0.0, 0.0

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

    def log_train(self):
        wandb.log({'train loss': self.train_loss, 'train accuracy': self.train_accuracy})

    def log_valid(self):
        wandb.log({'valid loss': self.valid_loss, 'valid accuracy': self.valid_accuracy})

    def process_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()

        for waveforms in loader:
            with torch.no_grad():
                waveforms = waveforms.to(self.params.device)
                melspecs = self.spectrogramer(waveforms)

                mu_law = self.quantizer(waveforms)
                zeros = torch.zeros((mu_law.shape[0], 1)).to(self.params.device)
                inputs = torch.cat([zeros, waveforms[:, :-1]], dim=1)
                targets = self.quantizer.quantize(mu_law[:, 1:])

            with torch.set_grad_enabled(train):
                self.optimizer.zero_grad()
                logits = self.model(inputs, melspecs)
                targets = targets[:, :logits.shape[-1]]
                loss = self.criterion(logits, targets)
                accuracy = (torch.argmax(logits, dim=1) == targets).to(torch.float).mean()

            if train:
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss.item()
                self.train_accuracy += accuracy.item()

                self.train_step += 1
                if self.train_step % self.params.log_steps == 0:
                    self.train_loss /= self.params.log_steps
                    self.train_accuracy /= self.params.log_steps
                    if self.params.use_wandb:
                        self.log_train()
                    self.train_loss, self.train_accuracy = 0.0, 0.0

            else:
                self.valid_loss += loss.item() * waveforms.shape[0]
                self.valid_accuracy += accuracy.item() * waveforms.shape[0]

        if not train:
            self.valid_loss /= len(loader.dataset)
            self.valid_accuracy /= len(loader.dataset)
            if self.params.use_wandb:
                self.log_valid()
            self.valid_loss, self.valid_accuracy = 0.0, 0.0

    def train(self, train_loader, valid_loader):
        for epoch in range(self.params.start_epoch, self.params.start_epoch + self.params.num_epochs):
            self.process_epoch(train_loader, train=True)
            self.process_epoch(valid_loader, train=False)
            self.save_checkpoint(epoch)
