import torch
import tqdm as tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import f1, accuracy
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertConfig
from datasets import ClassifierDataset

class ClassifierTrainingModel(pl.LightningModule):

    def __init__(self, batch_size, patience, load_pretrained=True):
        super().__init__()
        if load_pretrained is True:
            # init with pretrained weights
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', return_dict=True, num_labels=2) 
        else:
            # init weights randomly
            config = DistilBertConfig.from_pretrained('distilbert-base-cased', return_dict=True, num_labels=2)
            self.model = DistilBertForSequenceClassification(config)
        self.batch_size = batch_size
        self.patience = patience
        
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    def get_loader(self, split, shuffle):
        dataset = ClassifierDataset(split)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=dataset.collate_batch, num_workers=2)
        return dataloader

    def train_dataloader(self):
        return self.get_loader('train', True)

    def val_dataloader(self):
        return self.get_loader('val', False)

    def test_dataloader(self):
        return self.get_loader('test', False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2.65e-5, weight_decay=4e-3)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.patience)
        return {
            'optimizer' : optimizer,
            'scheduler' : scheduler,
            'monitor' : 'val_acc'
        }

    def training_step(self, batch, batch_nb):
        out = self(**batch)
        loss = out.loss
        return loss

    def validation_step(self, batch, batch_nb):
        out = self(**batch)
        return {
            'logits' : out.logits,
            'labels' : batch['labels']
        }

    def validation_epoch_end(self, outputs):
        logits = torch.cat([elem['logits'] for elem in outputs], 0)
        labels = torch.cat([elem['labels'] for elem in outputs], 0)
        pred = logits.softmax(-1).argmax(-1)
        acc = accuracy(pred, labels)
        micro_f1 = f1(pred, labels, 2, average='micro')
        macro_f1 = f1(pred, labels, 2, average='macro')
        self.log('val_acc', acc, prog_bar=True)
        self.log('micro_f1', micro_f1)
        self.log('macro_f1', macro_f1)

    def test_step(self, batch, batch_nb):
        out = self(**batch)
        return {
            'logits' : out.logits,
            'labels' : batch['labels']
        }
    
    def test_epoch_end(self, outputs):
        logits = torch.cat([elem['logits'] for elem in outputs], 0)
        labels = torch.cat([elem['labels'] for elem in outputs], 0)
        pred = logits.softmax(-1).argmax(-1)
        acc = accuracy(pred, labels)
        micro_f1 = f1(pred, labels, 2, average='micro')
        macro_f1 = f1(pred, labels, 2, average='macro')
        self.log('val_acc', acc, prog_bar=True)
        self.log('micro_f1', micro_f1)
        self.log('macro_f1', macro_f1)
        print('Accuracy:', acc)
        print('Micro F1:', micro_f1)
        print('Macro F1:', macro_f1)
        

def run_training():
    batch_size = 42
    patience = 2
    model = ClassifierTrainingModel(batch_size, patience)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='pl_checkpoints',
        filename='model-{epoch:02d}-{val_acc:.4f}',
        mode='max'
    )
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback],
        gpus=1
    )
    trainer.fit(model)

def run_test():
    trainer = pl.Trainer(gpus=1)
    model = ClassifierTrainingModel(42, 2)
    model.model.load_state_dict(torch.load('classifier.pt'))
    trainer.test(model)

run_test()