import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64
context_len = 128 # block_size
lr = 3e-4
epochs = 8000 # max_iters
n_embd = 128 #32 # how to determine that?
n_heads = 4
n_layer = 4
dropout = 0.2 # understand that!

# useful stuff
eval_interval = 300 # every how many iters should I estimate the loss?
eval_iters = 200 # estimate over how many recent iters?
device = 'cuda' if torch.cuda.is_available() else 'cpu' # not available :((
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with open("pan_tadeusz.txt", "r") as f: # no encoding="utf-8" when pan_tadeusz?!?!?!
    text = f.read()

# create a vocab (all the unique chars in the text)
chars = sorted(set(text)) # list
vocab_size = len(chars)
# create mappings
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x] # encoder: take a string, output a list of ints
decode = lambda s: "".join([itos[c] for c in s]) # decoder: take a list of ints, output a string

# let's now encode our text and store it in a vector
data = torch.tensor(encode(text), dtype=torch.long)

# train/val split
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batches(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - context_len, (batch_size,)) # batch_size numbers between 0 and len(data) - context_len
    x = torch.stack([data[i:i+context_len] for i in idx])
    y = torch.stack([data[i+1:i+context_len+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # does nothing here but good practice
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batches(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train() # does nothing here but good practice
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # why bias false?
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len))) # lower triangular matrix

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # x ---> "'private' information of a token: I'm a, say, 5th token and I have some identity. My information is kept in the vector x"
        # why does that work if x has 3 dims and key and query just 2? prob just ignores the 3rd one?
        k = self.key(x) # (B, T, C) ---> "here's what I'm interested in"
        q = self.query(x) # (B, T, C) ---> "here's what I have"
        wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # why the :T?
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C) ---> "if you find me interesting, here's what I will communicate to you"
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # WHAT

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #!!
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # understand that!
        self.ln2 = nn.LayerNorm(n_embd) # understand that!


    def forward(self, x):
        # residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# https://chat.openai.com/c/f1bc75e1-9725-452c-aba8-d1f38641688a !!!
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each letter in our vocab has a n_embd embedding vector of n_embd len
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # each each position in the context_len block also has a n_embd embedding vector of n_embd len
        self.position_embedding_table = nn.Embedding(context_len, n_embd)

        # # gives us the information about each letter in every context, taking previous letters into account
        # self.sa_heads = MultiHeadAttention(n_heads, n_embd // n_heads) # 4 heads of 8-dim self-attention
        # # linear layer with a non-linearity (no non-linear activation functions before)
        # self.ffwd = FeedForward(n_embd)

        # sa_heads and ffwd all at once + there can now be many layers
        # self.blocks = nn.Sequential(
        #     TransformerBlock(n_embd, 4),
        #     TransformerBlock(n_embd, 4),
        #     TransformerBlock(n_embd, 4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for _ in n_layer])
        self.ln = nn.LayerNorm(n_embd)
        # to compute actual logits for every letter in our vocab (vocab_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C), position embeddings for each letter in every batch
        x = tok_emb + pos_emb # (B, T, C), broadcasting
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss 

    def generate(self, idx, max_new_tokens):
        # now it's pretty bad, because we only use the last char but later it will change!
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_len:] # crop the context so that it's not bigger than context_len
            logits, _ = self(idx_crop) # get predictions (these embeddings) 
            logits = logits[:, -1, :] # get predictions only for the next character (thus -1) -> (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # plucks out index of the char (0-64) with the highest prob. index in the probs tensor corresponds to actual index in the emb table:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> "If input is a matrix with m rows, out is an matrix of shape (m×num_samples)."
            # append sampled index to the running sequence:
            idx = torch.cat((idx, idx_next), dim=1) # dim=1 -> concatenate "Time" -> (B, C) becomes (B, C+1)
        return idx


model = GPTLanguageModel() # vocab_size is now defined globally
m = model.to(device)

# FIND OUT WHAT'S THE DIFFERENCE BETWEEN SGD AND THAT!!!
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

for i in range(epochs):
    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batches('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward() # get gradients for all of the parameters
    optimizer.step() # update the parameters according to the gradient

init_idx = torch.zeros((1, 1), dtype=torch.long, device=device) # B=1, T=1 and it's holding a 0 (0 is a new line so a reasonable char to start with)
decoded = decode(model.generate(init_idx, max_new_tokens=500)[0].tolist()) # [0] -> 0th batch dimension
print(decoded)

# stats:

# 100000 epochs -> loss: 2.3590312004089355, 5000 epochs -> 2.6092894077301025 (before multihead sa or even earlier)

# batch_size = 32
# context_len = 8 
# lr = 1e-3
# epochs = 8000 
# n_embd = 32
# n_heads = 4

# multihead sa, no ffwd, no blocks - step 7800: train loss 2.2560, val loss 2.2644 // 0.009243 M parameters
# multihead sa,    ffwd, no blocks - step 7800: train loss 2.2362, val loss 2.2379 // 0.010299 M parameters
# multihead sa,    ffwd,  3 blocks - step 7800: train loss 2.2843, val loss 2.2801 // 0.018555 M parameters
# + residual connections           - step 7800: train loss 2.4118, val loss 2.4183 // 0.024891 M parameters
# + 4 x bigger hidden ffwd layer   - step 7800: train loss 2.4105, val loss 2.4182 // 0.043611 M parameters
# bad results -> n_embd = 256      - step 2100: train loss 2.4461, val loss 2.4521 // 2.412635 M parameters
# still bad -> n_embd=128, ctx=128 - step 1800: train loss 2.4054, val loss 2.4103 // 0.631899 M parameters
# ??? what the f, why so bad 
# batch_size = 64, lr = 3e-4       - step 600: train loss 2.4159, val loss 2.4214 // 0.631899 M parameters
# + layer norm                     - no difference // 0.633435 M parameters
# THE DUMBEST MISTAKE EVER in MultiHeadAttention, sth with x instead of out
# + layer norm and other prev stuff again -  // 0.633435 M parameters

# --- 

# BETTER INITIALIZATION!!! NOW IT'S TERRIBLE - watch makemore tutorial again to understand the weight init!
# CUDA!