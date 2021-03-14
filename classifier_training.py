import torch
import tqdm as tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional import f1
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import ClassifierDataset

class ClassifierTrainingModel(pl.LightningModule):

    def __init__(self, batch_size, patience):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', return_dict=True, num_labels=1) 
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
        optimizer = AdamW(self.parameters())
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
        pred = (logits.sigmoid() < 0.5).int().view(-1)
        acc = (pred == labels).sum() / len(pred)
        micro_f1 = f1(pred, labels, 1, average='micro')
        macro_f1 = f1(pred, labels, 1, average='macro')
        self.log('val_acc', acc, prog_bar=True)
        self.log('micro_f1', micro_f1)
        self.log('macro_f1', macro_f1)


def run_training():
    batch_size = 2
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

run_training()
