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

    def _extract_style(self, ids, mask):
        style_vec = self.style_encoder(
            input_ids=ids, attention_mask=mask, 
            output_hidden_states=True).last_hidden_state.mean(1)
        return style_vec
        

    def forward(self, input_ids, attention_mask, **inp):
        if input_ids.elems[-1] is True:
            context_ids, context_mask, input_ids, prefix, source_context_ids, source_context_mask, _ = input_ids
            target_vec = self._extract_style(context_ids, context_mask)
            source_vec = self._extract_style(source_context_ids, source_context_mask)
            input_vec = self._extract_style(input_ids, attention_mask[:, 1:])
            context_vec = input_vec + self.style_delta * (target_vec - source_vec)
        else:
            context_ids, context_mask, input_ids, prefix = input_ids
            context_vec = self._extract_style(context_ids, context_mask)
        encoding = self.encoder(input_ids=input_ids, 
                                attention_mask=attention_mask[:, 1:], **inp)
        encoding.last_hidden_state = encoding.last_hidden_state + context_vec[:, None, :]
        encoding.last_hidden_state = torch.cat([prefix[:, None, :], encoding.last_hidden_state], 1)
        return encoding

class DialogueRestyler(nn.Module):
    
    def __init__(self, apply_back_translation=False):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small',
                                                            return_dict=True)
        self.model.encoder = StyleEncoder(self.model.encoder)
        self.apply_back_translation=apply_back_translation

    def forward(self, context_ids, context_mask, input_ids, input_mask, target,
                prefix, source_context_ids=None, source_context_mask=None, inference=False):
        if self.apply_back_translation is True:
            context_ids, context_mask, input_ids, input_mask, target, prefix = \
                self.back_translation(context_ids, context_mask, input_ids, 
                                      input_mask, target, prefix)
        if inference is True:
            assert source_context_ids is not None and source_context_mask is not None
            input_ids = (context_ids, context_mask, input_ids, prefix, source_context_ids, source_context_mask)
        else:
            input_ids = (context_ids, context_mask, input_ids, prefix)
        out = self.model(input_ids=input_ids, attention_mask=input_mask, labels=target)
        return out
    
    def generate(self, context_ids, context_mask, input_ids, input_mask, prefix, *args,
                 inference=False, **kwargs):
      if inference is True:
        args = (*args, True)
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
      section_size = context.shape[0] // 4
      context_section = context[2*section_size:3*section_size, :]
      context_mask_section = context_mask[2*section_size:3*section_size, :]
      corrupted_section = corrupted[3*section_size:, :]
      corrupted_mask_section = corrupted_mask[3*section_size:, :]
      prefix_section = prefix[2*section_size:3*section_size, :]
      seq = self.generate(context_section, context_mask_section, corrupted_section, 
                  corrupted_mask_section, prefix_section, max_length=corrupted.shape[-1])
      pad = torch.zeros(seq.shape[0], corrupted.shape[1] - seq.shape[1], 
                      dtype=torch.long, device=seq.device)
      mask = torch.ones(seq.shape[0], seq.shape[1]+1, dtype=torch.long, device=seq.device)
      mask = torch.cat([mask, pad], 1)
      seq = torch.cat([seq, pad], 1)
      corrupted[3*section_size:, :] = seq
      corrupted_mask[3*section_size:, :] = mask
      return context, context_mask, corrupted, corrupted_mask, target, prefix

