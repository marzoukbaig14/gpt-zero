import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------
# hyperparameters
# ------------
batch_size = 32     # how many independent sequences to process in parallel
block_size = 8      # maximum context length for predictions (T)
max_iters = 3000    # total training iterations
eval_interval = 300 # how often to evaluate loss on train/val
eval_iters = 200    # how many batches to average loss over during evaluation
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# ------------
# data loading
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# ------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all unique characters in the text, this is our vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocabulary size: {vocab_size} unique characters")

# character level tokenizer: char -> int and int -> char
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]     # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # list of ints -> string

# encode the entire dataset into a 1D tensor of token indices
# shape: [len(text)] e.g. [1,115,394] for tinyshakespeare
data = torch.tensor(encode(text), dtype=torch.long)
print(f"dataset size: {len(data)} tokens")

# 90/10 train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------
# data batching
# ------------
def get_batch(split):
    # randomly sample batch_size starting positions in the dataset
    # each starting position gives us a sequence of block_size tokens
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x: input sequences,  shape: (B, T) = (batch_size, block_size)
    # y: target sequences (x shifted by 1), shape: (B, T) = (batch_size, block_size)
    # y[b, t] is always the character that follows x[b, t]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # move to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y

# ------------
# loss estimation
# ------------
@torch.no_grad() # no gradients needed for evaluation, saves memory
def estimate_loss():
    out = {}
    model.eval() # switch to eval mode (affects dropout, batchnorm etc)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # average loss over eval_iters batches
    model.train() # switch back to train mode
    return out

# ------------
# model
# ------------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # embedding table: shape (vocab_size, vocab_size) = (65, 65)
        # each row i contains the logits for what character comes after character i
        # this is the entire model — one lookup table, no hidden layers
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx:    (B, T) tensor of token indices, integers
        # targets:(B, T) tensor of target token indices, integers (optional)

        # lookup each token index in the embedding table
        # each integer -> its row in the table (a vector of vocab_size logit scores)
        # output shape: (B, T, C) where C = vocab_size = 65
        logits = self.token_embedding_table(idx)

        if targets is None:
            # inference mode, no loss needed
            loss = None
        else:
            # cross entropy expects (N, C) not (B, T, C)
            # so flatten B and T into one dimension
            # logits:  (B, T, C) -> (B*T, C)
            # targets: (B, T)    -> (B*T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # cross entropy internally applies softmax to logits
            # then computes negative log likelihood against the true target index
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) tensor of token indices, the current context
        # autoregressively generates max_new_tokens new tokens
        for _ in range(max_new_tokens):

            # forward pass to get logits for all positions
            # logits shape: (B, T, C)
            logits, loss = self(idx)

            # bigram model only uses the last token to predict the next
            # so slice off the last time step only
            # logits shape: (B, T, C) -> (B, C)
            logits = logits[:, -1, :]

            # softmax converts raw logit scores to a probability distribution
            # probs shape: (B, C), each row sums to 1
            probs = F.softmax(logits, dim=-1)

            # sample one token index from the probability distribution
            # idx_next shape: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append the new token to the running sequence
            # idx shape: (B, T) -> (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ------------
# training
# ------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)
print(f"model parameters: {sum(p.nelement() for p in m.parameters())}")

# AdamW optimizer: adaptive learning rates per parameter + weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # periodically evaluate and print train/val loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch
    # xb: (B, T), yb: (B, T)
    xb, yb = get_batch('train')

    # forward pass: compute logits and loss
    logits, loss = model(xb, yb)

    # backward pass: compute gradients and update weights
    optimizer.zero_grad(set_to_none=True) # clear gradients from previous step
    loss.backward()                        # compute gradients
    optimizer.step()                       # update weights

# ------------
# generation
# ------------
# start from a single zero token (the '.' padding character)
# shape: (1, 1) -> one sequence, one token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))