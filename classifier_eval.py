import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertConfig, logging
logging.set_verbosity_error()
import torch
import pandas as pd
from tqdm import tqdm

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
config = DistilBertConfig.from_pretrained('distilbert-base-cased', return_dict=True, num_labels=2)
model = DistilBertForSequenceClassification(config)
model.load_state_dict(torch.load('classifier.pt'))

data = pd.read_csv('dgst_eval.csv')
data = [data.iloc[i].tolist() for i in range(len(data))]
data = [(tokenizer(sent, return_tensors='pt').input_ids, tokenizer(pred, return_tensors='pt').input_ids if isinstance(pred, str) else None, sent, pred, score) for sent, pred, score in tqdm(data, desc='Preprocessing')]

result = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
model.to(device)

for sent_tokens, pred_tokens, sent, pred, score in tqdm(data, desc='Run predictions'):
    sent_class = model(sent_tokens.to(device)).logits[0].softmax(0).argmax().to(cpu)
    pred_class = model(pred_tokens.to(device)).logits[0].softmax(0).argmax().to(cpu) if pred_tokens is not None else None
    result.append((sent, pred, sent_class, pred_class, score))

data = pd.DataFrame(result, columns=['sentence', 'prediction', 'sentence_class', 'prediction_class', 'score'])
data.to_csv('dgst_full_eval.csv', index=False)
