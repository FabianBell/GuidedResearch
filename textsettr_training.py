import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

batch_size = 8
lr = 1e-3
num_epochs = 20
accumulate_grad = 10
save_every = 1000

def training():
    model = TextSETTR()

    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load(f'model.pt'))
        print('Model loaded')

    model.parallelize()

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
            out = model(*batch)
            loss = out.loss
            loss.backwards()
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
