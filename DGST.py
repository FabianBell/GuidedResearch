import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, EncoderDecoderModel

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

class DGST_Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-cased', return_dict=True)
        self.decoder = BertModel.from_pretrained('bert-base-cased', is_decoder=True, 
                                                 add_cross_attention=True, return_dict=True)
        self.lm_head = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, length=None):
        encoding = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = []
        inp = encoding.pooler_output.unsqueeze(1)
        for i in range(input_ids.shape[1] if length is None else length):
            out = self.decoder(inputs_embeds=inp, encoder_hidden_states=encoding.last_hidden_state,
                               encoder_attention_mask=attention_mask)
            out = out.last_hidden_state[:, -1, :].unsqueeze(1)
            outputs.append(out)
            inp = torch.cat([inp, out], dim=1)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.lm_head(outputs)
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
        self.t0 = DGST_LSTM()  #DGST_Transformer()
        self.t1 = DGST_LSTM()  #DGST_Transformer()
        self.vocab_size = vocab_size
        self.vocab_min = vocab_min
    
    def loss(self, data, target):
        loss = F.cross_entropy(data.view(-1, data.shape[-1]), target.view(-1))
        return loss
    
    def noise(self, data, p=0.03):
        data = data.clone()
        mask = torch.rand(*data.shape) < p
        data[mask] = torch.randint(self.vocab_min, self.vocab_size-1, (mask.sum(),), device=data.device)
        return data

    def forward(self, data0, data0_mask, data1, data1_mask):
        data0_n = self.noise(data0)
        data1_n = self.noise(data1)
       
        data0_1 = self.t1(data0, data0_mask)
        data0_1 = self.noise(data0_1.argmax(2))
        data0_1_0 = self.t0(data0_1)

        data1_0 = self.t0(data1, data1_mask)
        data1_0 = self.noise(data1_0.argmax(2))
        data1_0_1 = self.t1(data1_0) 

        l_t = self.loss(data0_1_0, data0) + self.loss(data1_0_1, data1)
        
        data0_n_0 = self.t0(data0_n, data0_mask)
        data1_n_1 = self.t1(data1_n, data1_mask)

        l_c = self.loss(data0_n_0, data0) + self.loss(data1_n_1, data1)

        loss = (l_t + l_c) / 4
        return ModelOutput(logits=data0_1, loss=loss)
