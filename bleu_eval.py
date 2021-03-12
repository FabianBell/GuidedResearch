import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import T5Tokenizer, logging, T5ForConditionalGeneration, T5Config
logging.set_verbosity_error()
import torch
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm import tqdm

tokenizer = T5Tokenizer.from_pretrained('t5-small')
config = T5Config.from_pretrained('t5-small')
model = T5ForConditionalGeneration(config)
model.load_state_dict(torch.load('dgst.pt'))

with open('response_dataset/test.txt', 'r') as dfile:
    data = dfile.read()

data = [(tokenizer(line, return_tensors='pt').input_ids, line) for line in tqdm(data.splitlines(), desc='Preprocessing')]
data = [(tokens, line) for tokens, line in data if len(tokens) <= 512]

result = []

for entry_tokens, entry in tqdm(data, desc='Run predictions'):
    predict_tokens = model.generate(entry_tokens, max_length=512)
    predict = tokenizer.decode(predict_tokens[0], skip_special_tokens=True)
    reference = entry.split(' ')
    hypothesis = predict.split(' ')
    score = sentence_bleu([reference], hypothesis)
    result.append((entry, predict, score))

data = pd.DataFrame(result, columns=['sentence', 'prediction', 'score'])
data.to_csv('eval.csv', index=False)
