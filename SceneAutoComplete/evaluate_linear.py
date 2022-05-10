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
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import math
import re
import pandas as pd
import ast 

# tensorboardX writer

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
        # encoders = []
        # for i in range(n_code):
        #     encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        # self.encoders = nn.ModuleList(encoders)

        #language model
        self.linear1 = nn.Linear(embed_size, embed_size)

        self.linear_dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.linear_dropout2 = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(seq_len)
        self.linear3 = nn.Linear(embed_size, embed_size)
        self.linear_dropout3 = nn.Dropout(dropout)
        self.predict = nn.Linear(embed_size, 2, bias=False)


    def forward(self, x):
        e = self.e_embeddings(x[:,:, 0])
        r = self.r_embeddings(x[:,:, 1])
        o = self.o_embeddings(x[:,:, 2])
        #combine
        x = self.combine(torch.cat([e, r, o], dim=-1))
        x = self.combine_dropout(x)

        x = self.linear1(x)
        x = self.linear_dropout1(x)
        x = self.linear2(x)
        x = self.linear_dropout2(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x).squeeze()
        # print(x.shape)

        x = self.linear3(x)
        x = self.linear_dropout3(x)
        pred = self.predict(x)
        # r_pred = self.predict_r(x)
        # o_pred = self.predict_o(x)
        return pred

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
        s, t, label = dataset.get_sentence_idx(index % len(dataset))
        len_s = len(s)
        # while len(s) < dataset.seq_len:
        #     s.extend()
        #     index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append([dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        
        # Add cls at start and sep at end
        # final_s = [[dataset.E_CLS_IDX, dataset.R_CLS_IDX, dataset.E_CLS_IDX]] + 
        final_s = s + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
        final_s = final_s + t #+ [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
        # print(final_s)
        return {'input': torch.Tensor([w for w in final_s]).long(),
                'target': label}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        # s = s.values
        context_triples = s[0]
        target_triples = s[1]
        label = s[2]
        context_triples = [t.split(',') for t in context_triples]
        try:
          target_triples = target_triples.split(',')
        except:
          print(index)
          exit()
        for i in range(len(context_triples)):
            if len(context_triples[i]) != 3:
                raise ValueError('triple not of size 3: {}'.format(s))
            assert isinstance(context_triples[i], list)
            assert isinstance(context_triples[i][0], str)
            assert isinstance(context_triples[i][1], str)
            assert isinstance(context_triples[i][2], str)
            
        s = []
        for k in context_triples:
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
        t = []
        try:
          t_e, t_r, t_o = target_triples
        except:
          print(index)
          exit()
        if t_e in dataset.e_vocab:
            t_e = dataset.e_vocab[t_e]
        else:
            # t_e = dataset.entity_OUT_OF_VOCAB_IDX
            raise ValueError('entity out of vocab: {}'.format(t_e))
        if t_r in dataset.r_vocab:
            t_r = dataset.r_vocab[t_r]
        else:
            # t_r = dataset.relation_OUT_OF_VOCAB_IDX
            raise ValueError('relation out of vocab: {}'.format(t_r))
        if t_o in dataset.e_vocab:
            t_o = dataset.e_vocab[t_o]
        else:
            # t_o = dataset.entity_OUT_OF_VOCAB_IDX
            raise ValueError('entity out of vocab: {}'.format(t_o))
        t.append([t_e, t_r, t_o])
        return s, t, label

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
train_text = json.load(open('val_preds_1.json', "r"))
# pth = 'lm_sentences.txt'
# sentences = open(pth).read().split('\n')
# sentences = [s for s in sentences if len(s) > 0]
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
dataset = SentencesDataset(train_text, e_vocab, r_vocab, seq_len)
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

print(len(dataset.e_vocab), len(dataset.r_vocab))
# =============================================================================
# Model
# =============================================================================
#init model
print('initializing model...')
model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)

# load model from pretrained path
pretrained_model = torch.load("./linear_model.pth")

pretrained_dict = pretrained_model.state_dict() #pretrained model keys
model_dict = model.state_dict() #new model keys

# import pdb; pdb.set_trace();

processed_dict = {}

for k in model_dict.keys(): 
    if(k in pretrained_dict):
        processed_dict[k] = pretrained_dict[k] 
    else:
        print("couldnt find layer: ", k)

model.load_state_dict(processed_dict, strict=False) 

# model.load_state_dict(torch.load('model_lm_epoch_1.pth'))
model = model.cuda()

# =============================================================================
# Eval
# =============================================================================
print('training...')
model.eval()
batch_iter = iter(data_loader)
n_iteration = len(data_loader)

preds = []
gts = []
for it in range(n_iteration):
    
    #get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)
    
    #infer
    input = batch['input']
    target = batch['target'].cuda(non_blocking=True)
    
    input = input.cuda(non_blocking=True)

    #forward
    output = model(input)
    pred_e = torch.argmax(output, axis=1)
    pred_e = pred_e.cpu().detach().numpy()
    gt = target.cpu().detach().numpy()

    preds += pred_e.tolist()
    gts += gt.tolist()


preds = np.array(preds)
gts = np.array(gts)

acc = (preds == gts).sum() / len(preds)
p = precision_score(gts, preds)
r = recall_score(gts, preds)
f = f1_score(gts, preds)


print("Total data points: ", len(preds))
print('acc: ', acc)
print('precision: ', p)
print('recall: ', r)
print('f1: ', f)
