nn
from transformers import T5ForConditionalGeneration

class Solist(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small',
                                                            return_dict=True)

    def forward(self, input_ids, input_mask, target):
        out = self.model(input_ids=input_ids, attention_mask=input_mask, labels=target)
        return out
    
    def generate(self, input_ids, input_mask, *args, **kwargs):
      pred = self.model.generate(input_ids=input_ids, 
                                 attention_mask=input_mask, **kwargs)
      return pred
