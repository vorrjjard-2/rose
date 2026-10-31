"""
Training script for my first iteration of an LLM.
"""

import torch as t
import torch.nn as nn
import torch.optim as optim

from transformer import Transformer
from data import  Tokenizer, GPTDataset

from torch.utils.data import Dataset, DataLoader, random_split
from yaml import safe_load

from tqdm import tqdm

from einops import rearrange, reduce, einsum

import numpy as np

with open('config.yml', 'r') as file:
    cfg = safe_load(file)

tokenizer = Tokenizer(cfg)
dataset = GPTDataset(cfg, 'datasets/tinyshakespeare_flat', tokenizer)

train_ds, val_ds, test_ds = random_split(dataset, lengths=[0.05, 0.1, 0.85])

train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=True)
test_dl = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

model = Transformer(cfg).to(cfg['device'])
optimizer = optim.Adam(model.parameters(), lr=float(cfg['lr']))
loss_fn = nn.CrossEntropyLoss()

### we might be ready chat
#
#
#
print('Starting training !')

epoch_bar = tqdm(range(cfg['epochs']), position=0)
train_bar = tqdm(train_dl, position=1)
val_bar = tqdm(val_dl, position=1)


for epoch in epoch_bar:
    model.train()
    print(f'Epoch {epoch}')
    for tokens, targets in train_bar:

        tokens, targets = tokens.to(cfg['device']), targets.to(cfg['device'])

        targets = targets.flatten()
        logits = rearrange(model(tokens)[:, :-1], 'b posn d_vocab -> (b posn) d_vocab') # size [batch, posn, d_vocab] as probabilities, last one is unverifiable

        loss = loss_fn(logits, targets)

        loss.backward() # feed the loss back (backprop)
        optimizer.step() # move all the parameters
        optimizer.zero_grad() # reset the optimizer
        train_bar.set_postfix({"loss": f"{loss:.4f}"})

    if epoch % 5 == 0:
        model.eval()
        val_losses = []
        for tokens, targets in val_bar:
            tokens, targets = tokens.to(cfg['device']), targets.to(cfg['device'])

            targets = targets.flatten()
            logits = rearrange(model(tokens)[:, :-1], 'b posn d_vocab -> (b posn) d_vocab')

            loss = loss_fn(logits, targets)
            val_losses.append(loss.detach().item())

        print(f'Val Loss : {np.mean(val_losses)}')
            # IMPLEMENT SAMPLING LATER
