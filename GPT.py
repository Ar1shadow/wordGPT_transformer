
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import math


# ------ Hyperparameters -----
batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # the maximum context length for one sequence
N_train = 0.9   # percentage of data to use for training
max_iters = 10000 # number of training iterations
eval_interval = 1000  # how often to evaluate the loss on train and val sets
learning_rate = 1e-3  # learning rate for optimization
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200  # number of iterations to estimate the loss on train and val sets

# transformer hyperparameters
num_head = 1
d_model = 32
d_k = d_model // num_head


# ----- prepare the dataset-----
with open('./input.txt', 'r') as f:
        text = f.read()
chars = sorted(list(set(text)))  # get all the unique characters that occur in this text
vocab_size = len(chars)    # the number of unique characters in the text, also the size of the vocabulary
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
data = [stoi[c] for c in text]  # encode the text into a list of integers
n = int(len(data) * N_train) # n was calculated as float!!!, it should be converted to integer
train_set = data[:n]
val_set = data[n:]
        
def encode(str : list[str]) -> list[int]:
        return [stoi[s] for s in str]

def decode(ints : list[int]) -> str:
        return ''.join([itos[i] for i in ints])

# simplified dataloader
def get_batch(split : str):
    data = train_set if split == 'train' else val_set
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random starting indices for the batch
    x = torch.stack([torch.tensor((data[i:i+block_size])) for i in ix])  # satck the input sequences up to a row, then 4 x 8 tensor
    y = torch.stack([torch.tensor((data[i+1:i+block_size+1])) for i in ix]) # +1 to shift the target sequence by one character, target is the next character in the sequence
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
             x, y = get_batch(split)
             logits = model(x)
             losses[k] = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        out[split] = losses.mean()
    model.train()
    return out
    

# ----- model definition -----
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.LM_head = nn.Linear(d_model, vocab_size)  # language model head to project the output of the transformer to the vocabulary size
        self.Q = nn.Linear(d_model, d_model, bias=False)  # Query : what i want to query
        self.K = nn.Linear(d_model, d_model, bias=False)  # Key : what i have
        self.V = nn.Linear(d_model, d_model, bias=False)  # Value : what i have that i can give to you
        self.d_k = d_k

        # precompute RoPE cos/sin as buffers so they follow .to(device) automatically
        theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k/2,) 每个纬度的频率不同，偶数维和奇数维共享频率
        angles = torch.outer(torch.arange(block_size).float(), theta)     # (block_size, d_k/2) 
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())
        
    def forward(self, idx):
        Batch_size, Time_steps = idx.shape
        vocab_emb = self.token_embedding_table(idx)  # (Batch_size, Time_steps, d_model)
        
        ''' single head self-attention '''
        
        # 2 : compute attention weights
        query = self.Q(vocab_emb)  # (B, T, d_model) vector representation of the queries
        key = self.K(vocab_emb)  # (B, T, d_model) vector representation of the keys
        value = self.V(vocab_emb)  # (B, T, d_model) vector representation of the values
        
        # 3 : positional encoding with RoPE
        Q_w = self.apply_rope(query)
        K_w = self.apply_rope(key)
        attn_weights = Q_w @ K_w.transpose(-2, -1) / math.sqrt(self.d_k) # (B, T, T) attention weights, row to col : row have how much attention to pay to col

        # casual masking : mask strictly upper triangle (future tokens), keep self+past
        mask = torch.triu(torch.ones(Time_steps, Time_steps, device=idx.device), diagonal=1).bool()  # this is a tensor, so need to specify device
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, T, T) normalize the attention weights
        out = torch.matmul(attn_weights, value)  # (B, T, d_model) weighted sum of the values based on the attention weights
        
        logits = self.LM_head(out)  # (Batch_size, Time_steps, vocab_size)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits = self(idx[:, -block_size:])  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C) the last time step
            probs = F.softmax(logits, dim=-1)  # (B, C) probabilities for the next token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to the running sequence
        return idx

    def apply_rope(self, x):
        """
        x: (B, T, d_k)  对每个位置的向量做旋转变换
        cos/sin 已在 __init__ 预计算为 buffer，自动跟随 device。
        """
        T = x.size(1)
        cos = self.rope_cos[:T][None, :, :]  # (1, T, d_k/2)
        sin = self.rope_sin[:T][None, :, :]

        x1 = x[..., 0::2]   # 偶数维
        x2 = x[..., 1::2]   # 奇数维

        # 旋转公式：[x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated




class MLVisualizer:
    def __init__(self, style='seaborn-v0_8-muted', font_size=10):
        plt.style.use(style)
        plt.rcParams.update({'font.size': font_size, 'figure.dpi': 120})
        self.primary_color = '#3498db'
        self.secondary_color = '#e74c3c'
    
    def plot_training(self, history : dict, save_path = None): 
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5)) 
        epochs = range(len(history['train_loss']))
        if history.get('train_loss') and history.get('val_loss'):
            pass
        else:
            print("No training history to plot.")
            return
        
        ax1.plot(epochs,history['train_loss'], label='Train Loss', color=self.primary_color)
        ax1.plot(epochs,history['val_loss'], label='Validation Loss', color=self.secondary_color, linestyle='--')        
        best = np.argmin(history['val_loss'])
        ax1.annotate(f'Best Loss: {history["val_loss"][best]:.4f}', 
                     xy=(best+1, history['val_loss'][best]),
                     xytext=(best+5, history['val_loss'][best]+0.2),
                     arrowprops=dict(arrowstyle='->', color='black'))
        
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='upper right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  
        


if __name__ == '__main__':
    gpt = GPT()
    gpt.to(device) 
    plt_visualizer = MLVisualizer()
    # preparation for training
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    criiterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': []}
    # -- training loop ---
    for steps in range(max_iters+1):
        gpt.train()
        x, y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True is more efficient than zero_grad()
        
        logits = gpt(x)
        loss = criiterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)  # clip the gradients to prevent exploding gradients
        
        loss.backward()  # backpropagate the gradients
        optimizer.step()
        scheduler.step()  # update the learning rate
        
        if steps % eval_interval == 0:
            losses = estimate_loss(gpt)
            print(f"step {steps} / {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            history['train_loss'].append(losses['train'].item())
            history['val_loss'].append(losses['val'].item())

    print(decode(gpt.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=300)[0].tolist()))
    plt_visualizer.plot_training(history, 'training_history_transformer_v1.png')

        
