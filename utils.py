import torch as t
import torch.nn as nn
import yaml
from yaml import safe_load
import datasets
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, cfg : dict):
        self.cfg = cfg
        self.possible_tokens = list(cfg['possible_tokens'])

        self.encoder = {char : i for i, char in enumerate(self.possible_tokens)}
        self.decoder = {i : char for i, char in enumerate(self.possible_tokens)}

    def encode(self, x):
        return [self.encoder[char] for char in x]

    def decode(self, x):
        return [self.decoder[char] for char in x]

if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        cfg = safe_load(file)

    tokenizer = Tokenizer(cfg)
