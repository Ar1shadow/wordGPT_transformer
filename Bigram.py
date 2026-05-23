import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
'''
using Bigram model, which actually is based on the Markov chain 1st order,the next character is only related to the current character
then move to the Transformer model
'''


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 8  # the maximum context length for one sequence
        self.batch_size = 32  # how many independent sequences will we process in parallel
        self.N_train = 0.9   # percentage of data to use for training, rest for validation
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device

    
    def dataloader(self, source_path : str):
        with open(source_path, 'r') as f:
            text = f.read()
        _,self.n_model = self.tokenizer(text)
        self.data = self.encode(text)
        n = int(len(self.data) * self.N_train) # n was calculated as float!!!, it should be converted to integer
        self.train_set = self.data[:n]
        self.val_set = self.data[n:]
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.n_model, self.n_model) 

    def get_batch(self, split : str):
        data = self.train_set if split == 'train' else self.val_set
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,)) # generate random starting indices for the batch
        x = torch.stack([torch.tensor((data[i:i+self.block_size])) for i in ix])  # satck the input sequences up to a row, then 4 x 8 tensor
        y = torch.stack([torch.tensor((data[i+1:i+self.block_size+1])) for i in ix]) # +1 to shift the target sequence by one character, target is the next character in the sequence
        return x, y


    def tokenizer(self,source_raw : list[str]) -> tuple[list[str], int, dict[str, int], dict[int, str]]: 
        chars = sorted(list(set(source_raw)))
        vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        return chars, vocab_size

    def encode(self, str : list[str]) -> list[int]:
        return [self.stoi[s] for s in str]

    def decode(self, ints : list[int]) -> str:
        return ''.join([self.itos[i] for i in ints])


    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (Batch_size, Time_steps, Channels)
        if targets is None: 
            loss = None
        else:
            # B, T, C -> (B*T, C), also for targets, B, T -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # logit is (B, T, C)
            logits = logits[:, -1, :]  # focus only on the last time step
            # prbs is (B, C)
            probs = F.softmax(logits, dim=-1)  # convert to probabilities on the last dimension : channels
            # (B ,1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from the distribution
            # (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to the running sequence
        return idx
    

if __name__ == '__main__':

    # ----  construction -----
    bigram = BigramLanguageModel()
    bigram.to(bigram.device)
    bigram.dataloader('./input.txt')


# ----  before training -----
    x, y = bigram.get_batch('train')
    print('x: ', x.shape, x)
    print('y: ', y.shape, y)
    logits, loss = bigram(x, y)
    print('logits: ', logits.shape)
    print('loss: ', loss)  # 4.6
    

# ---- training preparation -----
    print("parameter:", bigram.parameters())
    epochs = 10000
    optimizer  = torch.optim.AdamW(bigram.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ---- training loop -----
    bigram.train()
    for steps in range(epochs):
        x, y = bigram.get_batch('train') # fetch the batch
        x .to(bigram.device)    
        y .to(bigram.device)
        optimizer.zero_grad(set_to_none=True)  # reset the gradients

        logits, loss = bigram(x, y)  # forward pass
        loss.backward()  # backpropagation

        torch.nn.utils.clip_grad_norm_(             # 梯度裁剪，防止梯度爆炸
            bigram.parameters(), max_norm=1.0
        )

        optimizer.step()  # update the parameters
        
        scheduler.step()  # update the learning rate
        if steps % 1000 == 0:
            print(f"step {steps} / {epochs}, loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': bigram.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
}, 'bigram_model.pth')


idx = torch.zeros((1, 1), dtype=torch.long) # idx (B,T)  here start with zero '/n'
print(bigram.decode(bigram.generate(idx, max_new_tokens=100)[0].tolist())) # decode the generated sequence and print it out