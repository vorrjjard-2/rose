"""
Training script for my first iteration of an LLM.
"""

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from einops import einsum, rearrange, reduce
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from yaml import safe_load

from data import GPTDataset, Tokenizer
from transformer import Transformer


class Trainer:
    def __init__(self, cfg, model, dataset, checkpoint_path):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg

        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, lengths=[0.05, 0.1, 0.85]
        )

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.val_dl = DataLoader(
            self.val_ds, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.test_dl = DataLoader(
            self.test_ds, batch_size=self.cfg["batch_size"], shuffle=False
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.cfg["lr"]))

    def training_step(self):
        model.train()
        train_bar = tqdm(self.train_dl, position=1)

        train_losses = []

        for tokens, targets in train_bar:
            tokens, targets = (
                tokens.to(self.cfg["device"]),
                targets.to(self.cfg["device"]),
            )

            targets = targets.flatten()
            logits = rearrange(
                self.model(tokens)[:, :-1], "b posn d_vocab -> (b posn) d_vocab"
            )  # size [batch, posn, d_vocab] as probabilities, last one is unverifiable

            loss = self.loss_fn(logits, targets)

            loss.backward()  # feed the loss back (backprop)
            self.optimizer.step()  # move all the parameters
            self.optimizer.zero_grad()  # reset the optimizer
            train_bar.set_postfix({"loss": f"{loss:.4f}"})

            train_losses.append(loss.detach().item())

        return np.mean(train_losses)

    @t.no_grad()
    def validation_step(self):
        self.model.eval()
        val_bar = tqdm(self.val_dl, position=1)

        val_losses = []
        accs = []

        for tokens, targets in val_bar:
            tokens, targets = (
                tokens.to(self.cfg["device"]),
                targets.to(self.cfg["device"]),
            )

            targets = targets.flatten()
            logits = rearrange(
                self.model(tokens)[:, :-1], "b posn d_vocab -> (b posn) d_vocab"
            )

            loss = self.loss_fn(logits, targets)
            val_losses.append(loss.detach().item())

            predictions = t.argmax(
                logits, dim=-1
            ).long()  # now just shape (b posn), same as target
            accuracy = ((predictions == targets).sum() / len(targets)).item()

            accs.append(accuracy)

            val_loss, val_acc = np.mean(val_losses), np.mean(accs)

            val_bar.set_postfix(
                {"val_loss_avg": f"{val_loss:.4f}", "val_acc_avg": f"{val_acc:.4f}"}
            )

            return (val_loss, val_acc)

    def save_model(self, filename="model"):
        dir = self.cfg["checkpoint_dir"]

        t.save(self.model.state_dict(), f"{dir}/{filename}.pt")
        print(f"Model saved to {dir}/{filename}.pt")

    def train_model(self, verbose_losses=True):
        epoch_bar = tqdm(range(self.cfg["epochs"]), position=0)
        for epoch in epoch_bar:
            train_loss = self.training_step()

            if epoch % 5 == 0:
                val_loss, val_acc = self.validation_step()

            if (
                self.cfg["save_model_every"]
                and epoch % self.cfg["save_model_every"] == 0
            ):
                self.save_model()

            epoch_bar.set_postfix(
                {
                    "train_loss_avg": f"{train_loss:.4f}",
                    "val_loss_avg": f"{val_loss:.4f}",
                    "val_acc_avg": f"{val_acc:.4f}",
                }
            )

        self.save_model("final_model")
        print("All done!")

        tokenizer = Tokenizer(cfg)


if __name__ == "__main__":
    with open("config.yml", "r") as file:
        cfg = safe_load(file)

    tokenizer = Tokenizer(cfg)
    dataset = GPTDataset(cfg, "datasets/tinyshakespeare_flat", tokenizer)

    model = Transformer(cfg).to(cfg["device"])
    trainer = Trainer(cfg, model, dataset, "/checkpoints")

    ### Here we go

    trainer.train_model()
