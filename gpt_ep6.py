import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
context_len = 8 # block_size
lr = 1e-2
epochs = 5000 # max_iters
n_embd = 32 # how to determine that?

# useful stuff
eval_interval = 300 # every how many iters should I estimate the loss?
eval_iters = 200 # estimate over how many recent iters?
device = 'cuda' if torch.cuda.is_available() else 'cpu' # not available :((
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with open("input.txt", "r", encoding="utf-8") as f:
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

# most basic model - bigram
class BigramLanguageModel(nn.Module):
    def __init__(self): # "__init__() call to the parent class must be made before assignment on the child."
        super().__init__()
        
        # 65 x n_embd table
        # https://stackoverflow.com/questions/50747947/embedding-in-pytorch
        # "nn.Embedding holds a Tensor of dimension (vocab_size, vector_size), 
        # i.e. of the size of the vocabulary x the dimension of each vector embedding, and a method that does the lookup.
        # When you create an embedding layer, the Tensor is initialised randomly. 
        # It is only when you train it when this similarity between similar words should appear."
        
        # each token directly reads off the logits for the next token from a lookup table
        # these just become probabilities for all the 65 chars to be the next char, given some char
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # "A simple lookup table that stores embeddings of a fixed dictionary and size."
        self.position_embedding_table = nn.Embedding(context_len, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): # "Defines the computation performed at every call. Should be overridden by all subclasses."
        # idx and targets are both (B,T) tensor of integers
        B, T, C = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C), position embeddings for each letter in every batch
        x = tok_emb + pos_emb # (B, T, C) -> ?
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
            logits, _ = self(idx) # get predictions (these embeddings) 
            logits = logits[:, -1, :] # get predictions only for the next character (thus -1) -> (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # plucks out index of the char (0-64) with the highest prob. index in the probs tensor corresponds to actual index in the emb table:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> "If input is a matrix with m rows, out is an matrix of shape (m×num_samples)."
            # append sampled index to the running sequence:
            idx = torch.cat((idx, idx_next), dim=1) # dim=1 -> concatenate "Time" -> (B, C) becomes (B, C+1)
        return idx


model = BigramLanguageModel() # vocab_size is now defined globally
m = model.to(device)

# FIND OUT WHAT'S THE DIFFERENCE BETWEEN SGD AND THAT!!!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(epochs): # 100000 epochs -> loss: 2.3590312004089355, 5000 epochs -> 2.6092894077301025
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

print(loss.item())

init_idx = torch.zeros((1, 1), dtype=torch.long, device=device) # B=1, T=1 and it's holding a 0 (0 is a new line so a reasonable char to start with)
encoded = model.generate(init_idx, max_new_tokens=500)[0].tolist() # [0] -> 0th batch dimension
decoded = decode(encoded)
print(decoded) # still really bad, but at least there are some spaces

