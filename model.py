import torch.nn as nn
from transformers import T5ForConditionalGeneration

class Soloist(nn.Module):

    def __init__(self, dropout=0.3):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small',
                                                            return_dict=True)
        self.classifer = nn.Sequential(
            nn.Linear(self.model.config.d_model, self.model.config.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.model.config.d_model, 1),
            nn.Sigmoid()
        )
        self.classifier_criterion = nn.BCELoss()

    def forward(self, input_ids, target, corrupted_ids, corrupted_target):
        out = self.model(input_ids=input_ids, labels=target)
        embedding = self.model.encoder(input_ids=corrupted_ids).last_hidden_state[:, -1, :]
        pred = self.classifer(embedding)[:, 0]
        loss = self.classifier_criterion(pred, corrupted_target)
        return out.loss + loss
  
    def generate(self, input_ids, input_mask, *args, **kwargs):
      pred = self.model.generate(input_ids=input_ids, 
                                 attention_mask=input_mask, **kwargs)
      return pred
