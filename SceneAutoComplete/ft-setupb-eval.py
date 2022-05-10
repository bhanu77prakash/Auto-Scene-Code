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
import  tensorboardX as tbx
import ast 
import argparse

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
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)
        
        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, embed_size)
        self.linear_dropout = nn.Dropout(dropout)
        # self.predict = nn.Linear(embed_size, 2, bias=False)
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
        x = x[:, :, :]
        x = self.linear(x)
        x = self.linear_dropout(x)
        # pred = self.predict(x)
        r_pred = self.predict_r(x)
        o_pred = self.predict_o(x)
        return r_pred,o_pred

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
        s, label = dataset.get_sentence_idx(index % len(dataset))
        len_s = len(s)
        # while len(s) < dataset.seq_len:
        #     s.extend()
        #     index += 1
        
        #ensure that the sequence is of length seq_len
        # pid = [id for id, x in enumerate(s) if x[1] == dataset.R_MASK_IDX][0]
        # pre_context = np.random.randint(0, dataset.seq_len-2)
        # s = s[pid-pre_context:]
        s = s[:dataset.seq_len]
        len_s = len(s)
        [s.append([dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]) for i in range(dataset.seq_len - len_s)] #PAD ok
        
        #apply random mask
        
        # Add cls at start and sep at end
        final_s = [[dataset.E_CLS_IDX, dataset.R_CLS_IDX, dataset.E_CLS_IDX]] + s + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
        # final_s = final_s + t + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
        # print(final_s)
        return {'input': torch.Tensor([w for w in final_s]).long(),
                'target': torch.Tensor(label).long()}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        # s = s.values
        context_triples = s[0]
        # target_triples = s[1]
        label = s[1]
        context_triples = [t.split(',') for t in context_triples]
        # try:
        #   target_triples = target_triples.split(',')
        # except:
        #   print(index)
        #   exit()
        count = 0 
        for i in range(len(context_triples)):
            if len(context_triples[i]) != 3:
                assert len(context_triples[i]) == 1
                count += 1 
                # raise ValueError('triple not of size 3: {}'.format(s))
            else:
                assert isinstance(context_triples[i], list)
                assert isinstance(context_triples[i][0], str)
                assert isinstance(context_triples[i][1], str)
                assert isinstance(context_triples[i][2], str)

        assert count == 1   
        s = []
        flag = False
        for k in context_triples:
            try:
                e, r, o = k
                if e in dataset.e_vocab:
                    e = dataset.e_vocab[e]
                else:
                    # e = dataset.entity_OUT_OF_VOCAB_IDX
                    raise ValueError('entity out of vocab: {}'.format(e))
                if r in dataset.r_vocab:
                    r = dataset.r_vocab[r]
                else:
                    # r = dataset.relation_OUT_OF_VOCAB_IDX
                    # print(r)
                    raise ValueError('relation out of vocab: {}'.format(r))
                if o in dataset.e_vocab:
                    o = dataset.e_vocab[o]
                else:
                    # o = dataset.entity_OUT_OF_VOCAB_IDX
                    raise ValueError('entity out of vocab: {}'.format(o))
            except:
                e = k[0]
                if e in dataset.e_vocab:
                    e = dataset.e_vocab[e]
                else:
                    # e = dataset.entity_OUT_OF_VOCAB_IDX
                    raise ValueError('entity out of vocab: {}'.format(e))
                r = dataset.R_MASK_IDX
                o = dataset.E_MASK_IDX
                flag = True
            #   print(k)
              # exit()
            
            s.append([e, r, o])

        assert flag
        
        label = label.split(',') 
        label = [dataset.r_vocab[label[0]], dataset.e_vocab[label[1]]]
        # t = []
        # try:
        #   t_e, t_r, t_o = target_triples
        # except:
        #   print(index)
        #   exit()
        # if t_e in dataset.e_vocab:
        #     t_e = dataset.e_vocab[t_e]
        # else:
        #     # t_e = dataset.entity_OUT_OF_VOCAB_IDX
        #     raise ValueError('entity out of vocab: {}'.format(t_e))
        # if t_r in dataset.r_vocab:
        #     t_r = dataset.r_vocab[t_r]
        # else:
        #     # t_r = dataset.relation_OUT_OF_VOCAB_IDX
        #     raise ValueError('relation out of vocab: {}'.format(t_r))
        # if t_o in dataset.e_vocab:
        #     t_o = dataset.e_vocab[t_o]
        # else:
        #     # t_o = dataset.entity_OUT_OF_VOCAB_IDX
        #     raise ValueError('entity out of vocab: {}'.format(t_o))
        # t.append([t_e, t_r, t_o])
        return s, label

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
raw_train_text = json.load(open('../top_150_50_new/ft_test.json', "r"))

print("Data size before cleaning ", len(raw_train_text))
# Skip examples where the GT example lies outside the sequence length

train_text = []
for x in raw_train_text:
    if [id for id, y in enumerate(x[0]) if len(y.strip().split(",")) == 1][0] >= seq_len:
        continue

    train_text.append(x)

print("Data size after cleaning ", len(train_text))

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
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':False, 'pin_memory':True, 'batch_size':batch_size}
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

print(len(dataset.e_vocab), len(dataset.r_vocab))
# =============================================================================
# Model
# =============================================================================
#init model
print('initializing model...')
model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)

# load model from pretrained path
pretrained_model = torch.load("./ft-layout_model.pth")

pretrained_dict = pretrained_model.state_dict() #pretrained model keys
model_dict = model.state_dict() #new model keys

# import pdb; pdb.set_trace();

processed_dict = {}

for k in model_dict.keys(): 
    if(k in pretrained_dict):
        processed_dict[k] = pretrained_dict[k] 
    else:
        print(f"Couldn't copy {k} layer")

model.load_state_dict(processed_dict, strict=False) 

# model.load_state_dict(torch.load('model_lm_epoch_1.pth'))
model = model.cuda()

# =============================================================================
# Eval
# =============================================================================
print('Evaluating...')
batch_iter = iter(data_loader)
n_iteration = len(data_loader)
model.eval()
preds_e, preds_r = [], []
gts_e, gts_r = [], []
for it in range(n_iteration):
    
    #get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)
    
    #infer
    input = batch['input']
    target = batch['target'].cuda(non_blocking=True)
    
    input = input.cuda(non_blocking=True)

    #forward
    output = model(input)
    
    #compute the cross entropy loss 
    # import pdb; pdb.set_trace();

    output_v_r = output[0].view(-1,output[0].shape[-1])
    output_v_o = output[1].view(-1,output[1].shape[-1])

    target_r = target[:, 0]
    target_o = target[:, 1]

    mask_r = (input[:, :, 1].view(-1,1).squeeze() == dataset.R_MASK_IDX)
    mask_o = (input[:, :, 2].view(-1,1).squeeze() == dataset.E_MASK_IDX)

    # print(output_v_o.shape, mask_o.shape, mask_o.sum())

    # if mask_o.sum().item() != batch_size:
    #     mask = (input[:, :, 1] == dataset.R_MASK_IDX)
    #     # print(dataset.R_MASK_IDX, mask.shape)
    #     # print(mask[0])
    #     b_idx = torch.all(mask == False, axis=1).nonzero(as_tuple=False).item()
    #     print(it, b_idx)

    output_v_r = output_v_r[mask_r]
    output_v_o = output_v_o[mask_o]

    # print(output[0].shape, output_v_r.shape, output_v_o.shape, target_r.shape, target_o.shape, target_r.max(), target_r.min(), target_o.max(), target_o.min())
    
    
    # logger.add_scalar('train/entity/loss', e_loss.item(), it)
    # logger.add_scalar('train/relation/loss', r_loss.item(), it)
    # logger.add_scalar('train/loss', model_loss.item(), it)


    # compute predictoins using argmax

    # if masked_e.sum() > 0:
        # print("found a batch with masked elements")
    pred_e = torch.argmax(output_v_o, axis=1)
    pred_e = pred_e.cpu().detach().numpy()
    gt = target_o.cpu().detach().numpy()

    preds_e += pred_e.tolist()
    gts_e += gt.tolist()
    # pred_e = (pred_e > 0.5).astype(int)

    acc_e = (pred_e == gt).sum() / pred_e.shape[0]
    # p_e = precision_score(gt, pred_e, average="macro")
    # r_e = recall_score(gt, pred_e, average="macro")
    # f_e = f1_score(gt, pred_e, average="macro")

    pred_r = torch.argmax(output_v_r, axis=1)
    pred_r = pred_r.cpu().detach().numpy()
    gt = target_r.cpu().detach().numpy()
    # pred_e = (pred_e > 0.5).astype(int)
    preds_r += pred_r.tolist()
    gts_r += gt.tolist()

    acc_r = (pred_r == gt).sum() / pred_r.shape[0]
    # p_r = precision_score(gt, pred_r, average="macro")
    # r_r = recall_score(gt, pred_r, average="macro")
    # f_r = f1_score(gt, pred_r, average="macro")

    # log the accuracy of subject, relation and object predictions
    # logger.add_scalar('train/entity/acc', acc_e, it)
    # logger.add_scalar('train/entity/precision', p_e, it)
    # logger.add_scalar('train/entity/recall', r_e, it)
    # logger.add_scalar('train/entity/f1', f_e, it)

    # logger.add_scalar('train/relation/acc', acc_r, it)
    # logger.add_scalar('train/relation/precision', p_r, it)
    # logger.add_scalar('train/relation/recall', r_r, it)
    # logger.add_scalar('train/relation/f1', f_r, it)

    
    #print step
    print('it:', it, 
              ' | relation_acc',np.round(acc_r, 4),
            #   ' | relation_p',np.round(p_r, 4), 
            #   ' | relation_r',np.round(r_r, 4),
              ' | entity_acc',np.round(acc_e, 4),
            #   ' | entity_p',np.round(p_e, 4), 
            #   ' | entity_r',np.round(r_e, 4),
               )
    
    #reset gradients
    

# =============================================================================
# Results analysis
# =============================================================================

print("overall accuracy: ")
gts_e = np.array(gts_e)
gts_r = np.array(gts_r)
preds_e = np.array(preds_e)
preds_r = np.array(preds_r)
acc_e = (preds_e == gts_e).sum() / len(preds_e)
acc_r = (preds_r == gts_r).sum() / len(preds_r)

print('| relation_acc',np.round(acc_r, 4),
            #   ' | relation_p',np.round(p_r, 4), 
            #   ' | relation_r',np.round(r_r, 4),
              ' | entity_acc',np.round(acc_e, 4),
            #   ' | entity_p',np.round(p_e, 4), 
            #   ' | entity_r',np.round(r_e, 4),
               )

# print('end')
