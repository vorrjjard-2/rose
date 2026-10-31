import torch as t
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from yaml import safe_load
import einops
from einops import einsum, rearrange, repeat, reduce

class LNorm(nn.Module):
    """
    Normalises activations at a layer. It should have learned bias and weight.
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg['d_model']
        self.w, self.b = nn.Parameter(t.ones(d_model)), nn.Parameter(t.zeros(d_model))
        self.eps = 1e-5

    def forward(self, x): # x with shape batch, posn, d_model:
        ex = t.mean(x, dim=-1, keepdim=True)
        var = t.var(x, dim=-1, unbiased=False, keepdim=True) # should have same shape as above

        mid = (x - ex) / t.sqrt(var + self.eps)

        return (mid * self.w) + self.b

class Embedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        W_E_shape = (cfg['vocab_size'], cfg['d_model'])
        self.W_E = nn.Parameter(t.empty(size=W_E_shape)) # transforms token -> embed dim
        nn.init.normal_(self.W_E, std=0.2) # arbitrary value

    def forward(self, x):
        return self.W_E[x]

class PositionalEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_ctx, d_model = cfg['n_ctx'], int(cfg['d_model'])
        W_PE_shape = (n_ctx, d_model)
        W_PE = t.empty(size=W_PE_shape)


        positions = t.arange(0, n_ctx)[:, None]
        divisors = 10000 ** (2 * t.arange(0, d_model, 2) / d_model)

        mid = positions / divisors # broadcasting ! shape : n_ctx, d_model / 2 since the dims are interweaved

        W_PE[:,0::2] = t.sin(mid)
        W_PE[:,1::2] = t.cos(mid)

        self.register_buffer("W_PE", W_PE)

    def forward(self, x):
        return self.W_PE[x]


# ATTENTION

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init_range = cfg['init_range']

        d_model, n_heads = cfg['d_model'], cfg['n_heads']
        d_head = d_model // n_heads

        self.device = cfg['device']

        self.W_Q = nn.Parameter(t.empty(size=(n_heads, d_model, d_head)))
        self.W_K = nn.Parameter(t.empty(size=(n_heads, d_model, d_head)))
        self.W_V = nn.Parameter(t.empty(size=(n_heads, d_model, d_head)))
        self.b_Q = nn.Parameter(t.zeros(size=(n_heads, d_head)))
        self.b_K = nn.Parameter(t.zeros(size=(n_heads, d_head)))
        self.b_V = nn.Parameter(t.zeros(size=(n_heads, d_head)))

        self.W_O = nn.Parameter(t.empty(size=(n_heads, d_head, d_model))) # project back up?
        self.b_O = nn.Parameter(t.zeros(d_model)) #only d_model since this is added AFTER summing heads
        self.register_buffer('ignore', t.tensor(-1e5, dtype=t.float32)) # NEED : specify device?

        nn.init.normal_(self.W_Q, std=self.init_range)
        nn.init.normal_(self.W_K, std=self.init_range)
        nn.init.normal_(self.W_V, std=self.init_range)
        nn.init.normal_(self.W_O, std=self.init_range)

    def apply_causal_mask(self, x):
        """
        We will recieve a square matrix of attention scores. We need to mask the upper right triangle, perhaps offset by a diagonal.
        x.shape : [b, n_head, posn_Q, posn_K]

        mask all heads simultaneously, we define a multi-head mask first?
        """

        _, _, posn_Q, posn_K = x.shape


        mask = t.triu(t.ones((posn_Q, posn_K), device=self.device), diagonal=1).bool() # boolean mask
        return x.masked_fill(mask, self.ignore)

    def scale(self, x):

        d_model, n_heads = cfg['d_model'], cfg['n_heads']
        d_head = d_model // n_heads

        return x / (d_head ** 0.5)

    def forward(self, x):
        """
         x.shape : [b, seq_len, d_model]

        """

        q = einsum(x, self.W_Q, 'b posn d_model, n_heads d_model d_head -> b posn n_heads d_head') + self.b_Q
        k = einsum(x, self.W_K, 'b posn d_model, n_heads d_model d_head -> b posn n_heads d_head') + self.b_K
        v = einsum(x, self.W_V, 'b posn d_model, n_heads d_model d_head -> b posn n_heads d_head') + self.b_V

        attn_scores = einsum(q, k, 'b posn_Q n_heads d_head, b posn_K n_heads d_head -> b n_heads posn_Q posn_K') # do queries attend to keys? they do.
        attn_scaled = self.scale(attn_scores)
        attn_masked = self.apply_causal_mask(attn_scaled)

        attn_softmax = t.softmax(attn_masked, dim=-1)

        z = einsum(attn_softmax, v, 'b n_heads posnQ posnK, b posnK n_heads d_head -> b posnQ n_heads d_head')
        head_contributions_individual = einsum(z, self.W_O, 'b posnQ n_heads d_head, n_heads d_head d_model -> b posnQ n_heads d_model')

        head_sum = reduce(head_contributions_individual, 'b posnQ n_heads d_model -> b posnQ d_model', reduction='sum') + self.b_O

        return head_sum


class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg['d_model']

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, bias=True)
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module): # this is POST NORM
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LNorm(cfg)
        self.attention = Attention(cfg)
        self.ln2 = LNorm(cfg)
        self.ffn = FFN(cfg)

    def forward(self, x):
        x = self.ln1(self.attention(x) + x)
        x = self.ln2(self.ffn(x) + x)
        return x


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        W_U_shape = (cfg['d_model'], cfg['vocab_size'])
        self.W_U = nn.Parameter(t.empty(size=W_U_shape)) # transforms token -> embed dim
        nn.init.normal_(self.W_U, std=0.2) # arbitrary value

        self.b_U = nn.Parameter(t.zeros(cfg['vocab_size']))

    def forward(self, x):
        return einsum(x, self.W_U, 'batch posn d_model, d_model d_vocab -> batch posn d_vocab') + self.b_U


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = Embedding(cfg)
        self.pos_embedding = PositionalEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for layer in range(cfg['n_layers'])])
        self.ln = LNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        x = self.embedding(tokens) + self.pos_embedding(tokens)
        for layer in self.blocks :
            x = layer(x)

        x = self.ln(x)
        logits = self.unembed(x)

        return logits


if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        cfg = safe_load(file)

        tokens = t.ones((cfg['batch_size'], cfg['n_ctx'])).long()
        model = Transformer(cfg)

        out = model(tokens)

        print(out.shape)
