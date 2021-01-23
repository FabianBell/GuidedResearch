import pandas as pd
from torch.utils.data import Dataset
import os
import torch


DIALOGUE_PATH = 'dialogue_dataset'

class DialogueDataset(Dataset):

    def __init__(self, split):
        self.data = pd.read_json(os.path.join(DIALOGUE_PATH, f'{split}.json'))[['input', 'target']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        return entry.input, entry.target
    
    @staticmethod
    def _make_tensor(seq, pad):
        max_seq_len = max(map(len, seq))
        seq = [row + [pad]*(max_seq_len - len(row)) for row in seq]
        seq = torch.tensor(seq)
        return seq
        

    def collate_batch(self, batch):
        inp, tar = zip(*batch)
        inp = DialogueDataset._make_tensor(inp, 0)
        tar = DialogueDataset._make_tensor(tar, -100)
        return inp, tar
