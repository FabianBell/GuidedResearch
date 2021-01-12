import torch
import torch.nn as nn
import torch.nn.functional as F

class DGST(nn.Module):

    def __init__(self, vocab_size=28996, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, 4, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 4, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size*2, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, data):
        embedding = self.embedding(data)
        inp = embedding[:, :1, :]
        ctx = None
        outputs = []
        for i in range(data.shape[1]):
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
        self.t0 = DGST()
        self.t1 = DGST()
        self.vocab_size = vocab_size
        self.vocab_min = vocab_min
    
    def loss(self, data, target):
        loss = F.cross_entropy(data.view(-1, data.shape[-1]), target.view(-1), reduction="none")
        loss = loss.sum()
        return loss
    
    def noise(self, data, p=0.3):
        data = data.clone()
        mask = torch.rand(*data.shape) < p
        data[mask] = torch.randint(self.vocab_min, self.vocab_size-1, (mask.sum(),), device=data.device)
        return data

    def forward(self, data0, data1):
        data0_n = self.noise(data0)
        data1_n = self.noise(data1)
        
        data0_1 = self.t1(data0)
        data0_1 = self.noise(data0_1.argmax(2))
        data0_1_0 = self.t0(data0_1)

        data1_0 = self.t0(data1)
        data1_0 = self.noise(data1_0.argmax(2))
        data1_0_1 = self.t1(data1_0)

        l_t = self.loss(data0_1_0, data0) + self.loss(data1_0_1, data1)
        
        data0_n_0 = self.t0(data0_n)
        data1_n_1 = self.t1(data1_n)

        l_c = self.loss(data0_n_0, data0) + self.loss(data1_n_1, data1)

        loss = l_t + l_c
        return ModelOutput(logits=data0_1, loss=loss)
