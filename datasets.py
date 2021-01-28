import pandas as pd
import random
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import numpy as np
from collections import Counter
import os
import torch


STYLE_PATH = 'review_data'
DIALOGUE_PATH = 'dialogue_dataset'

CONTEXT_ID = 32109

class StyleDataset(Dataset):
    """
    Dataset for style based setence reconstruction
    """

    NOISES = ['drop', 'replace', 'shuffle']

    def __init__(self, split, dim=512, tuning_range=(0.2, 0.4), 
                 noises=['drop', 'replace']):
        super().__init__()
        assert noises is None or all([noise in self.NOISES for noise in noises])
        path = os.path.join(STYLE_PATH, f'{split}.pkl')
        self.data  = pd.read_pickle(path).Sentences.to_list() 
        self.dim = dim
        self.range = tuning_range
        self.noises = noises
    
    def noise(self, sample):
        prob = np.random.uniform(low=self.range[0], high=self.range[1], size=3)
        if self.NOISES[0] in self.noises:
          sample = self.drop_noise(prob[0], sample)
        if self.NOISES[1] in self.noises:
          sample = self.replace_noise(prob[1], sample)
        if self.NOISES[2] in self.noises:
          sample = self.shuffle_noice(prob[2], sample)
        return sample

    def drop_noise(self, p, sample):
        mask = np.random.rand(len(sample)) < p
        sample = [token for drop, token in zip(mask, sample) if drop.item() is False]
        return sample
    
    def replace_noise(self, p, sample):
        samples = random.choice(self.data)
        sample2 = random.choice(samples)
        mask = np.random.rand(len(sample)) < p
        sample = [sample2[idx] if idx < len(sample2) and replace.item() is True else token for idx, (replace, token) in enumerate(zip(mask, sample))]
        return sample
    
    def shuffle_noice(self, p, sample):
        mask = np.random.rand(len(sample)) < p
        selection = [elem for take, elem in zip(mask, sample) if take.item() is True]
        random.shuffle(selection)
        j = 0
        output = []
        for take, elem in zip(mask, sample):
            if take is True:
                output.append(selection[j])
                j += 1
            else:
                output.append(elem)
        return output

    def _get_attention_mask(self, seq_tensor):
        mask = (seq_tensor != 0).int()
        return mask

    def __len__(self):
        return len(self.data)
    
    def _get_ranges(self, center):
      width = np.random.rand()
      alignment = np.random.uniform(low=center-width, high=center+width)
      lower = max(alignment-width, 0)
      upper = min(alignment+width, 1)
      return lower, upper

    def _get_hidden_state_prefix(self, real, corrupted):
      real_counts = Counter(real)
      corrupted_counts = Counter(corrupted)
      relative = [real_counts.get(key, 0) - corrupted_counts.get(key, 0) for key in 
                  set(list(real_counts.keys()) + list(corrupted_counts.keys()))]
      add = -sum([rel for rel in relative if rel < 0])
      delete = sum([rel for rel in relative if rel > 0])
      add_rate = add / len(real)
      if len(corrupted) == 0:
        delete_rate = 1.0      
      else:
        delete_rate = delete / len(real)
      min_add, max_add = self._get_ranges(add_rate)
      min_del, max_del = self._get_ranges(delete_rate)
      prefix = np.zeros(self.dim)
      prefix[np.arange(4)] = [min_add, max_add, min_del, max_del]

      return prefix.tolist()

    def __getitem__(self, idx):
        sentences = self.data[idx]
        sample1, sample2 = [elem for elem in random.sample(sentences, k=2)]
        if self.noises is not None:
          corrupted = self.noise(sample2)
        else:
          corrupted = sample2
        prefix = self._get_hidden_state_prefix(sample2, corrupted)
        assert all([elem >= 0 and elem <= 1 for elem in prefix]), prefix
        corrupted = [CONTEXT_ID] + corrupted
        return sample1, corrupted, sample2, prefix
    
    def collate_batch(self, batch):
        context, corrupted, target, prefix = zip(*batch)

        context = self._pad_seq(context)
        corrupted = self._pad_seq(corrupted) 
        target = self._pad_seq(target)
        
        context_mask = self._get_attention_mask(context)
        corrupted_mask = self._get_attention_mask(corrupted)
        target = self._make_label(target)
        prefix = torch.tensor(prefix)
        # add zeros to corrupted mask for the prefix that is prepended 
        # to the encoder hidden states in the model
        prepend = torch.ones(corrupted_mask.shape[0], dtype=torch.long)
        corrupted_mask = torch.cat([prepend[:, None], corrupted_mask], 1)
        assert context.shape == context_mask.shape
        assert tuple(corrupted.shape) == (corrupted_mask.shape[0], corrupted_mask.shape[1]-1)

        return context, context_mask, corrupted, corrupted_mask, target, prefix
