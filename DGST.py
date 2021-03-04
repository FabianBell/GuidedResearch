import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DGST_LSTM(nn.Module):

    def __init__(self, vocab_size=28996, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, 4, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 4, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size*2, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, data, *args, length=None):
        embedding = self.embedding(data)
        _, ctx = self.encoder(embedding)
        inp = embedding[:, :1, :]
        outputs = []
        for i in range(data.shape[1] if length is None else length):
            out, ctx = self.decoder(inp, ctx)
            out = self.dense(out)
            outputs.append(out)
            inp = self.embedding(out.argmax(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs

class ModelOutput:
    """
    Model output wrapper
    """

    def __init__(self, **kwargs):
        self._elems = kwargs

    def __getattr__(self, name):
        return self._elems[name]

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __iter__(self):
        for elem in self._elems:
            yield elem

class DGSTPair(nn.Module):

    def __init__(self, vocab_size=28996, vocab_min=106):
        super().__init__()
        self.t0 = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t1 = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.vocab_size = vocab_size
        self.vocab_min = vocab_min
    
    def loss(self, data, target):
        loss = F.cross_entropy(data.view(-1, data.shape[-1]), target.view(-1))
        return loss
    
    def noise(self, data, p=0.3):
        data = data.clone()
        mask = torch.rand(*data.shape) < p
        data[mask] = torch.randint(self.vocab_min, self.vocab_size-1, (mask.sum(),), device=data.device)
        return data

    def forward(self, data0, data0_mask, data1, data1_mask):
        data0_label = data0.clone()
        data0_label[data0_label == 0] == -100
        data1_label = data1.clone()
        data1_label[data1_label == 0] == -100
        
        l_c_0 = self.t0(input_ids=self.noise(data0), attention_mask=data0_mask, labels=data0_label).loss
        data1_0 = self.t1.generate(input_ids=data1, attention_mask=data1_mask)[:, 1:]
        data1_0_mask = (data1_0.clone() != 0).int()

        l_c_1 = self.t1(input_ids=self.noise(data1), attention_mask=data1_mask, labels=data1_label).loss
        data0_1 = self.t0.generate(input_ids=data0, attention_mask=data0_mask)[:, 1:]
        data0_1_mask = (data0_1.clone() != 0).int()

        l_g_0 = self.t0(input_ids=self.noise(data0_1), attention_mask=data0_1_mask, labels=data0_label).loss
        l_g_1 = self.t1(input_ids=self.noise(data1_0), attention_mask=data1_0_mask, labels=data1_label).loss

        loss = l_c_0 + l_c_1 + l_g_0 + l_g_1

        return ModelOutput(logits=None, loss=loss)
