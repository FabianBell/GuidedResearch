import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import T5Tokenizer, logging, T5ForConditionalGeneration, T5Config
logging.set_verbosity_error()
import torch
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm import tqdm
import random
from model import TextSETTR
from torch.utils.data import Dataset, DataLoader

class EvalDataset(Dataset):

    def _preprocess(self, data, tag):
        data = [(self.tokenizer(line).input_ids, line) for line in tqdm(data.splitlines(), desc=tag)]
        data = [(tokens, line) for tokens, line in data if len(tokens) <= 512]
        return data

    def __init__(self):
        EOB = '<EOB>'
        BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
        KB_PREFIX = ' DB: '
        EOKB = '<EOKB>'
        QUERY = 'query'
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small',
            additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])
        
        with open('response_dataset/test.txt', 'r') as dfile:
            data = dfile.read()
        self.data = self._preprocess(data, 'Preprocessing 1/2')
        with open('donald_trump/test.txt', 'r') as dfile:
            ref_data = dfile.read()
        self.ref_data = self._preprocess(ref_data, 'Preprocessing 2/2')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (*self.data[idx], *random.choice(self.ref_data))

    def _make_tensor(self, seq):
        max_len = max(map(len, seq))
        seq = [elem + [0] * (max_len - len(elem)) for elem in seq]
        return torch.tensor(seq)
    
    def collate_batch(self, batch):
        entry, line, ref_entry, ref_line = zip(*batch)
        tokens = self._make_tensor(entry)
        mask = (tokens != 0).int()
        ref_tokens = self._make_tensor(ref_entry)
        ref_mask = (ref_tokens != 0).int()
        return tokens, mask, line, ref_tokens, ref_mask, ref_line

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

textsettr = TextSETTR(on_single_gpu=device)
textsettr.load_state_dict(torch.load('textsettr.pt'))
textsettr.to(device)

dataset = EvalDataset()
dataloader = DataLoader(dataset, batch_size=40, collate_fn=dataset.collate_batch, num_workers=2)

result = []


for entry_tokens, entry_mask, entry, ref_tokens, ref_mask, ref in tqdm(dataloader, desc='Run predictions'):
    prefix = torch.zeros(ref_tokens.shape[0], 512, device=device)
    prefix[:, [0,2]] = 0.2
    prefix[:, [1,3]] = 0.4
    mask_prefix = torch.ones(512, 1)
    predict_tokens = textsettr.generate(
        ref_tokens.to(device),
        ref_mask.to(device),
        entry_tokens.to(device),
        torch.cat([mask_prefix, entry_mask], -1, device=device),
        prefix,
        entry_tokens.to(device),
        entry_mask.to(device),
        max_length=512
    )
    predictions = dataset.tokenizer.batch_decode(predict_tokens.to(cpu), skip_special_tokens=True)
    for i, predict in enumerate(predictions):
        reference = entry[i].split(' ')
        hypothesis = predict.split(' ')
        score = sentence_bleu([reference], hypothesis)
        result.append((ref[i], entry[i], predict, score))

data = pd.DataFrame(result, columns=['reference', 'sentence', 'prediction', 'score'])
data.to_csv('eval.csv', index=False)
