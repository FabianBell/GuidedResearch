from transformers import T5TokenizerFast, logging
logging.set_verbosity_error()
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import sys
sys.path.append('..')

from datasets import DialogueDataset
from model import Soloist

dataset = DialogueDataset('val')

# create tokenizer
EOB = '<EOB>'
BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
KB_PREFIX = ' DB: '
EOKB = '<EOKB>'
QUERY = 'query'
tokenizer = T5TokenizerFast.from_pretrained('t5-small',
    additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])

# load model
model = Soloist()
model.load_state_dict(torch.load('model.pt'))

data = []

for i, (inp, target, _) in enumerate(tqdm(dataset, desc='Evaluate')):
    if i > 10:
        break
    inp_ids = tokenizer(inp, return_tensors='pt').input_ids
    pred = model.generate(inp_ids, max_length=512)
    out = tokenizer.decode(pred[0])
    hypothesis = out.split(' ')
    reference = target.split(' ')
    score = sentence_bleu([reference], hypothesis)
    data.append((inp, out, target, score))

df = pd.DataFrame(data, columns=['input', 'prediction', 'target', 'score'])
df.to_csv('eval.csv')
