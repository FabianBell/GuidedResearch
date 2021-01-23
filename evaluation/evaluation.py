from transformers import T5Tokenizer, logging
logging.set_verbosity_error()
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from datasets import DialogueDataset
from model import DialogueRestyler

dataset = DialogueDataset('test')

# create tokenizer
EOB = '<EOB>'
BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
KB_PREFIX = ' DB: '
EOKB = '<EOKB>'
QUERY = 'query'
tokenizer = T5Tokenizer.from_pretrained('t5-small',
    additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])

# load model
model = DialogueRestyler()
model.load_state_dict(torch.load('prod_model.pt'))

data = []

for inp_ids, target_ids in tqdm(dataset, desc='Evaluate'):
    model_inp = torch.tensor([inp_ids])
    pred = model.generate(
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        model_inp,
        torch.ones(model_inp.shape[0], model_inp.shape[1] + 1, dtype=torch.long),
        torch.zeros(1, 512),
        max_length=512
    )
    out = tokenizer.decode(pred[0])
    target = tokenizer.decode(target_ids)
    inp = tokenizer.decode(inp_ids)
    hypothesis = out.split(' ')
    reference = target.split(' ')
    score = sentence_bleu([reference], hypothesis)
    data.append((inp, out, target, score))

df = pd.DataFrame(data, columns=['input', 'prediction', 'target', 'score'])
df.to_csv('eval.csv')
