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

EOB = '<EOB>'
BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
KB_PREFIX = ' DB: '
EOKB = '<EOKB>'
QUERY = 'query'
tokenizer = T5Tokenizer.from_pretrained('t5-large',
    additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])
textsettr = TextSETTR()
textsettr.load_state_dict(torch.load('textsettr.pt'))

with open('response_dataset/test.txt', 'r') as dfile:
    data = dfile.read()

data = [(tokenizer(line, return_tensors='pt').input_ids, line) for line in tqdm(data.splitlines(), desc='Preprocessing 1/2')]
data = [(tokens, line) for tokens, line in data if len(tokens) <= 512]

with open('donald_trump/test.txt', 'r') as dfile:
    ref_data = dfile.read()

ref_data = [(tokenizer(line, return_tensors='pt').input_ids, line) for line in tqdm(ref_data.splitlines(), desc='Preprocessing 2/2')]
ref_data = [(tokens, line) for tokens, line in ref_data if len(tokens) <= 512]

inp_pipeline = map(lambda inp: (*random.choice(ref_data), *inp), data)

result = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
textsettr.to(device)

for ref_tokens, ref, entry_tokens, entry in tqdm(inp_pipeline, total=len(data), desc='Run predictions'):
    predict_tokens = textsettr.generate(
        ref_tokens.to(device),
        torch.ones(*ref_tokens.shape, dtype=torch.long, device=device),
        entry_tokens.to(device),
        torch.ones(entry_tokens.shape[0], entry_tokens.shape[1] + 1, dtype=torch.long, device=device),
        torch.tensor([[0.2, 0.4, 0.2, 0.4] + [0.]* 1020], device=device),
        entry_tokens.to(device),
        torch.ones(*entry_tokens.shape, dtype=torch.long, device=device),
        max_length=512
    )
    predict = tokenizer.decode(predict_tokens[0].to(cpu), skip_special_tokens=True)
    reference = entry.split(' ')
    hypothesis = predict.split(' ')
    score = sentence_bleu([reference], hypothesis)
    result.append((ref, entry, predict, score))

data = pd.DataFrame(result, columns=['reference', 'sentence', 'prediction', 'score'])
data.to_csv('eval.csv', index=False)
