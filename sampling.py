import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from yaml import safe_load

from data import Tokenizer
from transformer import Transformer


@t.no_grad()
def sample(
    model,
    cfg,
    tokenizer,
    prompt="I am a helpful assistant who",
    tokens_to_append=64,
    temperature=1.0,
):
    model.eval()
    tokens = tokenizer.encode(prompt.lower()).unsqueeze(0)
    n_ctx = cfg["n_ctx"]
    for _ in range(tokens_to_append):
        context = tokens if tokens.shape[-1] + 1 < n_ctx else tokens[:, -n_ctx:]
        logits = model(context)[:, -1, :] / temperature
        top_k = t.topk(logits[0], k=20)

        logits[logits < top_k.values[-1]] = float("-inf")

        softmax = nn.Softmax(dim=-1)

        probs = softmax(logits)
        token = t.multinomial(probs, num_samples=1)

        tokens = t.cat([tokens, token], dim=-1)

    output = tokenizer.decode(tokens.squeeze())

    return "".join(output)


if __name__ == "__main__":
    print("Hello!")

    with open("config.yml", "r") as file:
        cfg = safe_load(file)
    tokenizer = Tokenizer(cfg)
    model = Transformer(cfg)
    state_dict = t.load(
        "checkpoints/model.pt", map_location=t.device("cpu"), weights_only=True
    )
    model.load_state_dict(state_dict)

    print(sample(model=model, cfg=cfg, tokenizer=tokenizer))
