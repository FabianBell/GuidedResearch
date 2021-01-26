import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from transformers import T5Tokenizer
import random

DIALOGUE_PATH = 'dialogue_dataset'

EOB = '<EOB>'
BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
KB_PREFIX = ' DB: '
EOKB = '<EOKB>'
QUERY = 'query'

class DialogueDataset(Dataset):

    def __init__(self, split):
        self.data = pd.read_json(os.path.join(DIALOGUE_PATH, f'{split}.json'))[['input', 'target', 'segments']]
        self.tokenizer = tokenizer = T5Tokenizer.from_pretrained('t5-small',
            additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', 
            '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        return entry.input, entry.target, list(entry.segments.values())
    
    def collate_batch(self, batch):
        inp, tar, keys = zip(*batch)
        input_ids = self.tokenizer(list(inp), padding=True, return_tensors='pt').input_ids
        target_ids = self.tokenizer(list(tar), padding=True, return_tensors='pt').input_ids
        target_ids[target_ids == 0] = -100
        assert len(inp) % 2 == 0, 'Batch size has to be even'
        corrupted_size = len(inp) // 2
        # since task 1 and task 2 is independent, the tasks occure equally often 
        # and the corruption propaility is 1/3 for every corruption type, we can 
        # simply corrupt half of the batch
        corrupted = list(inp)[:corrupted_size]
        corrupted_target = [0] * corrupted_size
        replacements = []
        for i in range(corrupted_size):
            replacements.extend(keys[i])
        for i in range(corrupted_size):
            replaced = False
            seq = tar[i+corrupted_size]
            for key in keys[i+corrupted_size]:
                # there should almost always be a possible replacements
                if random.random() < 0.5 and len(replacements) > 0:
                    seq = seq.replace(key, random.choice(replacements))
                    replaced = True
            corrupted.append(inp[i+corrupted_size] + seq)
            corrupted_target.append(1 if replaced is True else 0)
        corrupted_target = torch.tensor(corrupted_target).float()
        corrupted_ids = self.tokenizer(corrupted, padding=True, return_tensors='pt').input_ids
        if corrupted_ids.shape[-1] > 512:
            # cannot resonably filter the data that could yield sequences that have over 512 tokens here
            # instead we will just clip the data 
            corrupted_ids = corrupted_ids[:, :512]
        return input_ids, target_ids, corrupted_ids, corrupted_target
