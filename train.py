"""
Training script for my first iteration of an LLM.
"""

import torch as t
import torch.nn as nn

from transformer import Transformer
from data import  Tokenizer, GPTDataset

print('hello')
