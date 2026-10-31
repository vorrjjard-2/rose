import torch as t
from torch.utils.data import Dataset, DataLoader
import datasets
from datasets import load_dataset



class Tokenizer:
    def __init__(self, cfg : dict):
        self.cfg = cfg
        self.possible_tokens = list(cfg['possible_tokens']) + ["'"]

        self.encoder = {char : i for i, char in enumerate(self.possible_tokens)}
        self.decoder = {i : char for i, char in enumerate(self.possible_tokens)}

    def encode(self, x):
        return t.tensor([self.encoder[char] for char in x], dtype=t.int)

    def decode(self, x):
        return [self.decoder[char.item()] for char in x]


class GPTDataset(Dataset):
    def __init__(self, cfg, dataset_name, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.vocab = set(cfg['possible_tokens'])
        with open(f'{dataset_name}.txt', 'r', encoding='utf-8') as file:
            dataset = file.read().lower()
            dataset_clean = [char for char in dataset if char in self.vocab]

        self.tokenizer = tokenizer

        self.dataset_tokenized = self.tokenizer.encode(dataset_clean)

    def __getitem__(self, idx):
        n_ctx = self.cfg['n_ctx']

        tokens = self.dataset_tokenized[idx:idx+n_ctx]
        return tokens

    def __len__(self):
        return len(self.dataset_tokenized)

if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        cfg = safe_load(file)

    tokenizer = Tokenizer(cfg)
    print(tokenizer.encode('hello world'))

    dataset = GPTDataset(cfg, 'datasets/tinyshakespeare_flat', tokenizer)
    print(dataset[0].shape)
