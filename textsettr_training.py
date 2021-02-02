import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

from model import TextSETTR
from datasets import StyleDataset

class Logger:

    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(f'{self.desc} ...', end='\r')
        return self

    def __exit__(self, exic_type, exc_value, exc_tb):
        print(f'{self.desc} [Done]')

batch_size = 6
num_epochs = 20
accumulate_grad = 20
save_every = 1000
lr = 1e-3 / accumulate_grad

from time import sleep

def training():
    with Logger('Initialise model'):
        model = TextSETTR()

        if os.path.exists('model.pt'):
                model.load_state_dict(torch.load(f'model.pt'))

        model.parallelize()

    with Logger('Load data'):
        dataset = StyleDataset('train', dim=model.model.config.d_model)
        dataloader = DataLoader(dataset, batch_size=batch_size,
            shuffle=True, collate_fn=dataset.collate_batch,
            num_workers=4, drop_last=True)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        loader = tqdm(dataloader, desc=f'Epoch {epoch}')
        losses = []
        for i, batch in enumerate(loader):
            
            # compute forward and backward pass
            try:
                loss = model(*batch)
            except RuntimeError:
                print('Batch skiped due to OOM')
                continue
            loss.backward()
            losses.append(loss.item())

            if i % accumulate_grad == 0 and i != 0:
                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                mean_loss = sum(losses) / len(losses)
                losses.clear()
                loader.set_postfix({'loss' : mean_loss})

            if i % save_every == 0 and i != 0:
                # save model
                torch.save(model.state_dict(), 'model.pt')
                print('Model saved')

if __name__ == '__main__':
    training()
