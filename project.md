Here’s the order I’d build it in. Get each piece working, and check its tensor shapes, before you start the next.

Data

Vocab and encoding. Build the sorted character set, then two mappings: stoi (character to integer) and itos (integer to character). Write encode and decode, and check that decode(encode(s)) == s.
Tensor and split. Encode the full text into one 1D long tensor, then cut it 90/10 into train and val.
Dataset and DataLoader. This is the sliding-window __getitem__ from earlier. Pull one batch and confirm x and y are both (B, T), with y shifted one position.

Model (shapes in brackets: B = batch, T = sequence length, C = embedding size)

Embeddings. One table for tokens, one for positions. Add them together to get (B, T, C).
Single attention head. Compute Q, K, V, scale the scores by √head_size, apply the causal mask, softmax, then multiply by V. Test this piece by itself.
Multi-head attention. Run several heads in parallel, concatenate them, then apply an output projection.
MLP. Linear layer to 4C, GELU, linear layer back to C.
Block. Pre-norm residuals, as in GPT-2: x + attn(ln(x)), then x + mlp(ln(x)).
Full model. Embeddings, then N blocks, a final LayerNorm, and a linear head out to vocab size. The output logits are (B, T, vocab).
Loss. Flatten the logits to (B·T, vocab) and the targets to (B·T), then apply cross-entropy.

Before you train, two sanity checks

Initial loss. An untrained model should give a loss of about ln(vocab_size). For 64 characters that’s about 4.16. A much higher number means your initialization or wiring is off.
Overfit one batch. Train on a single batch until the loss gets close to 0. If it can’t, there’s a bug somewhere, and running the full dataset won’t fix it.

Training and using it

Training loop. AdamW, then each step: forward pass, loss, zero_grad, backward, step. Every few hundred steps, measure val loss under torch.no_grad() with the model in eval() mode.
Generation. Crop the context to the last block_size tokens, take the logits at the last position, softmax, sample with torch.multinomial, append the sample, and repeat.

Start small: 2 to 4 layers, C = 64 to 128, T = 64. Make it correct first and scale it up after.

Most first-time bugs are a wrong causal mask (the model peeks at future tokens, so the loss falls suspiciously fast) or a forgotten model.train() after evaluation. If your val loss looks too good to be true, check the mask first.
