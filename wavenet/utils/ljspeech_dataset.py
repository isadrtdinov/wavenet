import random
import torch
import torchaudio


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, labels, params):
        super(LJSpeechDataset, self).__init__()

        self.labels = labels
        self.data_root = params.data_root
        self.audio_length = params.audio_length
        self.sample_rate = params.sample_rate

    def pad_sequence(self, sequence, max_length, fill=0.0, dtype=torch.float):
        padded_sequence = torch.full((max_length, ), fill_value=fill, dtype=dtype)
        sequence_length = min(sequence.shape[0], max_length)
        padded_sequence[:sequence_length] = sequence[:sequence_length]
        return padded_sequence

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        audio_info = self.labels.loc[index]
        waveform, sample_rate = torchaudio.load(self.data_root + audio_info.path + '.wav')

        if sample_rate != self.sample_rate:
            raise ValueError('Wrong sample rate!')

        waveform = waveform.view(-1)
        if waveform.shape[0] < self.audio_length:
            waveform = self.pad_sequence(waveform, self.audio_length)
        else:
            rand_index = random.randrange(waveform.shape[0] - self.audio_length)
            waveform = waveform[rand_index:rand_index + self.audio_length]

        return waveform
