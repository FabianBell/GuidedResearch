import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from copy import deepcopy

class TrojanHorse:
  
  def __init__(self, *elems):
    self.elems = elems
  
  def __getattr__(self, name):
    return self.elems[0].__getattribute__(name)

  def __iter__(self):
    for elem in self.elems:
        yield elem

class StyleEncoder(nn.Module):

    def __init__(self, encoder, style_delta=8):
        super().__init__()
        self.encoder = encoder
        self.style_encoder = deepcopy(encoder)
        self.style_delta = style_delta
        self.device = torch.device('cpu')
        self.first_device = torch.device('cpu')
        self.encoder.first_device = torch.device('cpu')

    def _extract_style(self, ids, mask):
        style_vec = self.style_encoder(
            input_ids=ids, attention_mask=mask, 
            output_hidden_states=True).last_hidden_state.mean(1)
        return style_vec

    def parallelize(self, device_map):
        return self.encoder.parallelize(device_map)

    def parallelize_style_encoder(self, device_map):
        self.style_encoder.parallelize(device_map)
        self.device = torch.device('cuda:0')
        self.first_device = self.encoder.first_device

    def forward(self, input_ids, attention_mask, **inp):
        attention_mask = attention_mask.to(self.encoder.first_device)
        input_ids = list(input_ids)
        if len(input_ids) == 6:
            target_ids, target_mask, input_ids, prefix, source_ids, source_mask = input_ids
            target_vec = self._extract_style(target_ids, target_mask)
            source_vec = self._extract_style(source_ids, source_mask)
            input_vec = self._extract_style(input_ids, attention_mask[:, 1:])
            context_vec = input_vec + self.style_delta * (target_vec - source_vec)
            encoding = self.encoder(input_ids=input_ids, 
                                    attention_mask=attention_mask[:, 1:], **inp)
            encoding.last_hidden_state += context_vec[:, None, :]
        else:
            context_ids, context_mask, input_ids, prefix = input_ids
            context_ids = context_ids.to(self.device)
            context_mask = context_mask.to(self.device)
            context_vec = self._extract_style(context_ids, context_mask)
            input_ids = input_ids.to(self.encoder.first_device)
            encoding = self.encoder(input_ids=input_ids, 
                                    attention_mask=attention_mask[:, 1:])
            encoding.last_hidden_state += context_vec[:, None, :].to(encoding.last_hidden_state.device)
        prefix = prefix.to(encoding.last_hidden_state.device)
        encoding.last_hidden_state = torch.cat([prefix[:, None, :], encoding.last_hidden_state], 1)
        return encoding

class TextSETTR(nn.Module):

    def __init__(self, apply_back_translation=False):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small',
                                                            return_dict=True)
        self.model.encoder = StyleEncoder(self.model.encoder)
        self.apply_back_translation=apply_back_translation
        self.output_device = torch.device('cpu')

    def parallelize(self):
        """
        Parallelize the model across different devices
        """
        device_map = {
            1 : [0,1,2,3,4,5],
            2 : [6,7,8,9,10,11,12,13,14],
            3 : [15,16,17,18,19,20,21,22,23]
        }
        self.model.parallelize(device_map=device_map)
        device_map = {
            0 : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            1 : [17,18,19,20,21,22,23]
        }
        self.model.encoder.parallelize_style_encoder(device_map)
        self.output_device = torch.device('cuda:1')

    def deparalelize(self):
        """
        Collects all model parts and moves them back to the cpu
        """
        self.model.deparalelize()
        self.model.encoder.style_encoder.to('cpu')

    def set_style_level(self, level):
        self.model.encoder.style_delta = level

    def forward(self, context_ids, context_mask, input_ids, input_mask, target, prefix):
        if self.apply_back_translation is True:
            context_ids, context_mask, input_ids, input_mask, target, prefix = \
                self.back_translation(context_ids, context_mask, input_ids, 
                                      input_mask, target, prefix)
        input_ids = (context_ids, context_mask, input_ids, prefix)
        target = target.to(self.output_device)
        out = self.model(input_ids=input_ids, attention_mask=input_mask, labels=target)
        return out.loss
    
    def generate(self, context_ids, context_mask, input_ids, input_mask, prefix, *args, **kwargs):
      input_ids = TrojanHorse(context_ids, context_mask, input_ids, prefix, *args)
      pred = self.model.generate(input_ids=input_ids, 
                                 attention_mask=input_mask, **kwargs)
      return pred
    
    def back_translation(self, context, context_mask, corrupted, corrupted_mask, 
                         target, prefix):
      """
      Applies a translation based on a fake context from the upper part of the
      batch stack. Should be only used during training.
      """
      section_size = context.shape[0] // 2
      context_section = context[:section_size, :]
      context_mask_section = context_mask[:section_size, :]
      corrupted_section = corrupted[section_size:, :]
      corrupted_mask_section = corrupted_mask[section_size:, :]
      prefix_section = prefix[:section_size, :]
      seq = self.generate(context_section, context_mask_section, corrupted_section, 
                  corrupted_mask_section, prefix_section, max_length=corrupted.shape[-1])
      pad = torch.zeros(seq.shape[0], corrupted.shape[1] - seq.shape[1], 
                      dtype=torch.long, device=seq.device)
      mask = torch.ones(seq.shape[0], seq.shape[1]+1, dtype=torch.long, device=seq.device)
      mask = torch.cat([mask, pad], 1)
      seq = torch.cat([seq, pad], 1)
      corrupted[section_size:, :] = seq
      corrupted_mask[section_size:, :] = mask
      return context, context_mask, corrupted, corrupted_mask, target, prefix

