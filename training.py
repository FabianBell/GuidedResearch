import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from datasets import *
from model import *

class TrainingModel(pl.LightningModule):

  def __init__(self, dataset_con, model_con, lr, batch_size, patience=2):
    super(TrainingModel, self).__init__()
    self.model = model_con()
    self.dataset_con = dataset_con
    self.batch_size = batch_size
    self.lr = lr
    self.patience = patience
  
  def forward(self, *inp):
    out = self.model(*inp)
    return out
  
  def get_dataloader(self, split, shuffle):
    """
    Creates the dataloader for the given split (['train', 'val', 'test'])
    """
    dataset = self.dataset_con(split) 
    dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=shuffle, collate_fn=dataset.collate_batch, 
                            num_workers=4)
    return dataloader
  
  def load_pretrained(self):
    self.model.load_pretrained()
  
  def train_dataloader(self):
    return self.get_dataloader('train', True)

  def val_dataloader(self):
    return self.get_dataloader('val', False)
  
  def test_dataloader(self):
    return self.get_dataloader('test', False)

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True)
    return {
        'optimizer' : optimizer,
        'scheduler' : scheduler,
        'monitor'   : 'val_loss'
    }

  def base_step(self, batch):
    context, context_mask, corrupted, corrupted_mask, target, prefix = batch
    out = self(context, context_mask, corrupted, corrupted_mask, target, prefix)
    loss = out.loss
    return loss

  def training_step(self, batch, batch_nb):
    loss = self.base_step(batch)
    self.log('train_loss', loss)
    if batch_nb % 1000 == 0 and torch.distributed.get_rank() == 0:
        # save in master
        print('Save model')
        torch.save(self.model.state_dict(), 'model.pt')
    return loss
  
  def validation_step(self, batch, batch_nb):
    loss = self.base_step(batch)
    self.log('val_loss', loss)
    return loss

"""## Training"""

#@title #Hyperparameters 
def run_training():
  name = 'DialogueRestyler' #@param {type: "string"}
  lr = 1e-3 #@param
  optimize_every = 5#@param
  batch_size=4 #@param
  patience=0 #@param
  epochs = 100 #@param
  check_val_every_n_epoch = 0.5 #@param
  model_con = lambda: DialogueRestyler(apply_back_translation=True)
  dataset_con = lambda split: StyleDialogueDataset(split, dim=512)
  model = TrainingModel(dataset_con, model_con, lr=lr, 
                      batch_size=batch_size, patience=patience)
  if os.path.exists(f'model.pt'):
    print('Model loaded')
    model.model.load_state_dict(torch.load(f'model.pt'))
  trainer = pl.Trainer(
      max_epochs=epochs,
      check_val_every_n_epoch=check_val_every_n_epoch, 
      gpus=-1,
      accelerator='ddp',
      accumulate_grad_batches=optimize_every
      )
  trainer.fit(model)

run_training()
