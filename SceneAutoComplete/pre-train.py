# =============================================================================
# Libs
# =============================================================================
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
import  tensorboardX as tbx

# tensorboardX writer

logger = tbx.SummaryWriter()

# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
#        self.q_linear = nn.Linear(out_dim, out_dim)
#        self.k_linear = nn.Linear(out_dim, out_dim)
#        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        
        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
        
        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        #n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
        
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class Transformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_e_embeddings, n_r_embeddings, seq_len, dropout=.1):
        super().__init__()
        
        #model input
        self.r_embeddings = nn.Embedding(n_r_embeddings, embed_size)
        self.e_embeddings = nn.Embedding(n_e_embeddings, embed_size)
        self.o_embeddings = nn.Embedding(n_e_embeddings, embed_size)
        self.combine = nn.Linear(embed_size*3, embed_size)
        self.combine_dropout = nn.Dropout(dropout)
        # self.pe = PositionalEmbedding(embed_size, seq_len)
        
        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)
        
        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear_dropout = nn.Dropout(dropout)
        self.predict_e = nn.Linear(embed_size, n_e_embeddings, bias=False)
        self.predict_r = nn.Linear(embed_size, n_r_embeddings, bias=False)
        self.predict_o = nn.Linear(embed_size, n_e_embeddings, bias=False)
                
    
    def forward(self, x):
        e = self.e_embeddings(x[:,:, 0])
        r = self.r_embeddings(x[:,:, 1])
        o = self.o_embeddings(x[:,:, 2])
        #combine
        x = self.combine(torch.cat([e, r, o], dim=-1))
        x = self.combine_dropout(x)

        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        x = self.linear_dropout(x)
        e_pred = self.predict_e(x)
        r_pred = self.predict_r(x)
        o_pred = self.predict_o(x)
        return [e_pred, r_pred, o_pred]

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len
    
# =============================================================================
# Dataset
# =============================================================================
class SentencesDataset(Dataset):
    #Init dataset
    def __init__(self, sentences, e_vocab, r_vocab, seq_len):
        dataset = self
        
        dataset.sentences = sentences
        print(len(e_vocab))
        
        dataset.e_vocab = e_vocab + ['<e_ignore>', '<e_oov>', '<e_mask>', '<e_cls>', '<e_sep>']
        dataset.e_vocab = {e:i for i, e in enumerate(dataset.e_vocab)} 
        dataset.e_rvocab = {v:k for k,v in dataset.e_vocab.items()}

        print(len(dataset.e_vocab))
        dataset.r_vocab = r_vocab + ['<r_ignore>', '<r_oov>', '<r_mask>', '<r_cls>', '<r_sep>']
        dataset.r_vocab = {e:i for i, e in enumerate(dataset.r_vocab)} 
        dataset.r_rvocab = {v:k for k,v in dataset.r_vocab.items()}
        dataset.seq_len = seq_len
        
        #special tags
        dataset.E_CLS_IDX = dataset.e_vocab['<e_cls>']
        dataset.E_SEP_IDX = dataset.e_vocab['<e_sep>']
        dataset.E_IGNORE_IDX = dataset.e_vocab['<e_ignore>'] #replacement tag for tokens to ignore
        dataset.E_OUT_OF_VOCAB_IDX = dataset.e_vocab['<e_oov>'] #replacement tag for unknown words
        dataset.E_MASK_IDX = dataset.e_vocab['<e_mask>'] #replacement tag for the masked word prediction task
        
        dataset.R_CLS_IDX = dataset.r_vocab['<r_cls>']
        dataset.R_SEP_IDX = dataset.r_vocab['<r_sep>']
        dataset.R_IGNORE_IDX = dataset.r_vocab['<r_ignore>'] #replacement tag for tokens to ignore
        dataset.R_OUT_OF_VOCAB_IDX = dataset.r_vocab['<r_oov>'] #replacement tag for unknown words
        dataset.R_MASK_IDX = dataset.r_vocab['<r_mask>'] #replacement tag for the masked word prediction task
    
    
    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self
        
        #while we don't have enough word to fill the sentence for a batch
        s = dataset.get_sentence_idx(index % len(dataset))
        len_s = len(s)
        # while len(s) < dataset.seq_len:
        #     s.extend()
        #     index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append([dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        final_s = []
        for i in range(len(s)):
            w = s[i]
            if i >= len_s:
                final_s.append((w, [dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]))
            else:
                if random.random() < p_random_mask:
                    final_s.append(([dataset.E_MASK_IDX, dataset.R_MASK_IDX, dataset.E_MASK_IDX], w))
                else:
                    e_random = random.random()
                    r_random = random.random()
                    o_random = random.random()
                    if e_random < p_random_mask:
                        e_in, e_out = dataset.E_MASK_IDX, w[0]
                    else:
                        e_in, e_out = w[0], dataset.E_IGNORE_IDX
                    if r_random < p_random_mask:
                        r_in, r_out = dataset.R_MASK_IDX, w[1]
                    else:
                        r_in, r_out = w[1], dataset.R_IGNORE_IDX
                    if o_random < p_random_mask:
                        o_in, o_out = dataset.E_MASK_IDX, w[2]
                    else:
                        o_in, o_out = w[2], dataset.E_IGNORE_IDX
                    final_s.append(([e_in, r_in, o_in], [e_out, r_out, o_out]))
        # s = [([dataset.E_MASK_IDX, dataset.R_MASK_IDX, dataset.E_MASK_IDX], w) if random.random() < p_random_mask else (w, [dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]) for w in s]
        
        # Add cls at start and sep at end
        final_s = [([dataset.E_CLS_IDX, dataset.R_CLS_IDX, dataset.E_CLS_IDX], [dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX])] + final_s + [([dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX], [dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX])]
        
        return {'input': torch.Tensor([w[0] for w in final_s]).long(),
                'target': torch.Tensor([w[1] for w in final_s]).long()}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        triples = s.split('\t')
        triples = [t.split('[DELIMITER]') for t in triples]
        for i in range(len(triples)):
            if len(triples[i]) != 3:
                raise ValueError('triple not of size 3: {}'.format(s))
            assert isinstance(triples[i], list)
            assert isinstance(triples[i][0], str)
            assert isinstance(triples[i][1], str)
            assert isinstance(triples[i][2], str)
            
        s = []
        for k in triples:
            try:
              e, r, o = k
            except:
              print(k)
              # exit()
            if e in dataset.e_vocab:
                e = dataset.e_vocab[e]
            else:
                # e = dataset.entity_OUT_OF_VOCAB_IDX
                raise ValueError('entity out of vocab: {}'.format(e))
            if r in dataset.r_vocab:
                r = dataset.r_vocab[r]
            else:
                # r = dataset.relation_OUT_OF_VOCAB_IDX
                print(r)
                raise ValueError('relation out of vocab: {}'.format(r))
            if o in dataset.e_vocab:
                o = dataset.e_vocab[o]
            else:
                # o = dataset.entity_OUT_OF_VOCAB_IDX
                raise ValueError('entity out of vocab: {}'.format(o))
            s.append([e, r, o])
        return s

# =============================================================================
# Methods / Class
# =============================================================================
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

# =============================================================================
# #Init
# =============================================================================
print('initializing..')
batch_size = 64
seq_len = 64
embed_size = 128
inner_ff_size = embed_size * 4
n_heads = 8
n_code = 8
n_vocab = 40000
dropout = 0.3
n_workers = 12

#optimizer
optim_kwargs = {'lr':2e-3, 'weight_decay':1e-4, 'betas':(.9,.999)}

# =============================================================================
# Input
# =============================================================================
#1) load text
print('loading text...')
pth = 'lm_sentences.txt'
sentences = open(pth).read().split('\n')
sentences = [s for s in sentences if len(s) > 0]
#2) tokenize sentences (can be done during training, you can also use spacy udpipe)
# print('tokenizing sentences...')
# sentences = [[w.strip().split(",")[1] for w in s.strip().split("\t") if len(w)] for s in sentences]

#3) create vocab if not already created
print('creating/loading vocab...')
pth = 'lm_relations.txt'
r_vocab = open("lm_relations.txt").read().split('\n')
r_vocab = [r for r in r_vocab if len(r) > 0]

e_vocab = open("lm_objects.txt").read().split('\n')
print(len(e_vocab))
e_vocab = [r for r in e_vocab if len(r) > 0]
print(len(e_vocab))

#4) create dataset
print('creating dataset...')
dataset = SentencesDataset(sentences, e_vocab, r_vocab, seq_len)
kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

print(len(dataset.e_vocab), len(dataset.r_vocab))
# =============================================================================
# Model
# =============================================================================
#init model
print('initializing model...')
model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)
model = model.cuda()

# =============================================================================
# Optimizer
# =============================================================================
print('initializing optimizer and loss...')
optimizer = optim.Adam(model.parameters(), **optim_kwargs)
r_loss_model = nn.CrossEntropyLoss(ignore_index=dataset.R_IGNORE_IDX)
e_loss_model = nn.CrossEntropyLoss(ignore_index=dataset.E_IGNORE_IDX)
o_loss_model = nn.CrossEntropyLoss(ignore_index=dataset.E_IGNORE_IDX)

# =============================================================================
# Train
# =============================================================================
print('training...')
print_each = 500
model.train()
batch_iter = iter(data_loader)
n_iteration = 10000
for it in range(n_iteration):
    
    #get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)
    
    #infer
    masked_input = batch['input']
    masked_target = batch['target']
    
    masked_input = masked_input.cuda(non_blocking=True)
    masked_target_e = masked_target[:, :, 0].cuda(non_blocking=True)
    masked_target_r = masked_target[:, :, 1].cuda(non_blocking=True)
    masked_target_o = masked_target[:, :, 2].cuda(non_blocking=True)

    #forward
    output = model(masked_input)
    
    #compute the cross entropy loss 
    output_v_e = output[0].view(-1,output[0].shape[-1])
    output_v_r = output[1].view(-1,output[1].shape[-1])
    output_v_o = output[2].view(-1,output[2].shape[-1])

    target_v_e = masked_target_e.view(-1,1).squeeze()
    target_v_r = masked_target_r.view(-1,1).squeeze()
    target_v_o = masked_target_o.view(-1,1).squeeze()
    
    e_loss = e_loss_model(output_v_e, target_v_e)
    r_loss = r_loss_model(output_v_r, target_v_r)
    o_loss = o_loss_model(output_v_o, target_v_o)


    loss = (e_loss + r_loss + o_loss) / 3
    #compute gradients
    loss.backward()
    
    #apply gradients
    optimizer.step()

    # log the three losses and the overall loss
    logger.add_scalar('train/subject_loss', e_loss.item(), it)
    logger.add_scalar('train/relation_loss', r_loss.item(), it)
    logger.add_scalar('train/object_loss', o_loss.item(), it)
    logger.add_scalar('train/loss', loss.item(), it)

    # compute accuracy of subject, relation and object predictiions for the non-masked elements

    masked_e = (masked_input[:, :, 0].view(-1,1).squeeze() == dataset.E_MASK_IDX)
    masked_r = (masked_input[:, :, 1].view(-1,1).squeeze() == dataset.R_MASK_IDX)
    masked_o = (masked_input[:, :, 2].view(-1,1).squeeze() == dataset.E_MASK_IDX)

    # compute predictoins using argmax

    # if masked_e.sum() > 0:
        # print("found a batch with masked elements")
    pred_e = output_v_e.argmax(dim=1)
    pred_r = output_v_r.argmax(dim=1)
    pred_o = output_v_o.argmax(dim=1)

    e_acc = (pred_e[masked_e] == target_v_e[masked_e]).sum().item() / float(masked_e.sum().item())
    r_acc = (pred_r[masked_r] == target_v_r[masked_r]).sum().item() / float(masked_r.sum().item())
    o_acc = (pred_o[masked_o] == target_v_o[masked_o]).sum().item() / float(masked_o.sum().item())

    # log the accuracy of subject, relation and object predictions
    logger.add_scalar('train/subject_acc', e_acc, it)
    logger.add_scalar('train/relation_acc', r_acc, it)
    logger.add_scalar('train/object_acc', o_acc, it)

    
    #print step
    if it % print_each == 0:
        print('it:', it, 
              ' | loss', np.round(loss.item(),2),
              ' | Δw:', round(model.r_embeddings.weight.grad.abs().sum().item(),3), 
              ' | Δr:', round(model.e_embeddings.weight.grad.abs().sum().item(),3))
    
    #reset gradients
    optimizer.zero_grad()
    

# =============================================================================
# Results analysis
# =============================================================================

torch.save(model, './pre_trained_model.pth')
# print('saving embeddings...')
# N = 3000
# np.savetxt('values.tsv', np.round(model.r_embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
# s = [dataset.rvocab[i] for i in range(N)]
# open('names.tsv', 'w+').write('\n'.join(s) )


# print('end')
