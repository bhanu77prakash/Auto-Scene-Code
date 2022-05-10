# =============================================================================
# Libs
# =============================================================================
from lib2to3.pgen2.token import NEWLINE
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

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, default='transformer', help='Model to train')
parser.add_argument('--mode', type=str, default='eval', help='mode of the model')
parser.add_argument('--save_probs', type=str, default=None, help='Save output probabilities')
parser.add_argument('--data', type=str, default=None, help='data to evaluate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--model_path', type=str, default='finetuned_model.pth', help='path to the model')
parser.add_argument('--pad', type=bool, default=True, help='pad evaluation inputs  or not')

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

class CNN(nn.Module):
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
        self.cnn = nn.Conv2d(1, embed_size, (5, embed_size), stride=1)

        self.maxpool = nn.MaxPool1d(seq_len - 4)
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

        x = x.unsqueeze(dim=1)
        x = self.cnn(x)

        x = x.squeeze()
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        x = self.maxpool(x).squeeze()
        # print(x.shape)

        x = self.linear3(x)
        x = self.linear_dropout3(x)
        pred = self.predict(x)
        # r_pred = self.predict_r(x)
        # o_pred = self.predict_o(x)
        return pred


class LSTM(nn.Module):
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

        self.lstm = nn.LSTM(embed_size, embed_size, num_layers=2)
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

        x = self.lstm(x)[0][:,-1,:]

        # print(x.shape)

        x = self.linear3(x)
        x = self.linear_dropout3(x)
        pred = self.predict(x)
        # r_pred = self.predict_r(x)
        # o_pred = self.predict_o(x)
        return pred

class Linear(nn.Module):
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
        self.predict = nn.Linear(embed_size, 2, bias=False)
        # self.predict_r = nn.Linear(embed_size, n_r_embeddings, bias=False)
        # self.predict_o = nn.Linear(embed_size, n_e_embeddings, bias=False)
                
    
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
        x = x[:, 0, :]
        x = self.linear(x)
        x = self.linear_dropout(x)
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
    def __init__(self, sentences, e_vocab, r_vocab, seq_len, special_tokens=True):
        dataset = self
        self.special_tokens = special_tokens 
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
        if args.batch_size != 1 or args.pad == True:
            [s.append([dataset.E_IGNORE_IDX, dataset.R_IGNORE_IDX, dataset.E_IGNORE_IDX]) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        
        # Add cls at start and sep at end
        if self.special_tokens:
            final_s = [[dataset.E_CLS_IDX, dataset.R_CLS_IDX, dataset.E_CLS_IDX]] + s + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
            final_s = final_s + t + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]]
        else:
            final_s = s + [[dataset.E_SEP_IDX, dataset.R_SEP_IDX, dataset.E_SEP_IDX]] + t
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


def evaluate(saved_model, dataloader, model, samples):
    

    pretrained_dict = saved_model.state_dict() #pretrained model keys
    model_dict = model.state_dict() #new model keys

    processed_dict = {}

    for k in model_dict.keys(): 
        if(k in pretrained_dict):
            processed_dict[k] = pretrained_dict[k] 
        else:
            print("couldnt find layer: ", k)

    model.load_state_dict(processed_dict, strict=False) 
    model = model.cuda()

    # =============================================================================
    # Eval
    # =============================================================================
    print('training...')
    print("Total length of batches: ", len(dataloader))
    model.eval()
    batch_iter = iter(data_loader)
    n_iteration = len(data_loader)

    preds = []
    gts = []
    probs = []
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

        # apply softmax and get prob
        output = F.softmax(output, dim=1)
        probs += output.cpu().detach().numpy()[:, 1].tolist()
        preds += pred_e.tolist()
        gts += gt.tolist()


    preds = np.array(preds)
    probs = np.array(probs)
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
    if args.save_probs:
        assert len(samples) == len(probs)
        with open(args.save_probs, 'w') as f:
            for i in range(len(samples)):
                f.write(" ".join(samples[i][0]) + "\t" + samples[i][1] + '\t' + str(probs[i]) + '\n')
    return preds, gts

# =============================================================================
# Input
# =============================================================================
#1) load text
args = parser.parse_args()
pth = 'lm_relations.txt'
r_vocab = open("lm_relations.txt").read().split('\n')
r_vocab = [r for r in r_vocab if len(r) > 0]

e_vocab = open("lm_objects.txt").read().split('\n')
e_vocab = [r for r in e_vocab if len(r) > 0]

if args.data:
    files = [args.data]
else:
    files = ["val_preds_{missing}.txt".format(missing=missing) for missing in [1, 2, 3, 5]]

for file in files:

    print("*******************************")
    print("Evaluating on file: ", file)
    print("*******************************")
    train_text = json.load(open(file, "r"))
    #4) create dataset
    # import pdb; pdb.set_trace()
    print('creating dataset...')
    print("Evaluating for: ", args.model)
    isTransformer = False
    if args.model == 'transformer':
        isTransformer = True
    dataset = SentencesDataset(train_text, e_vocab, r_vocab, seq_len, special_tokens=isTransformer)
    kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    pretrained_model = torch.load(f"{args.model_path}")
    model_cls = {"lstm": LSTM, "transformer": Transformer, "cnn": CNN, "linear": Linear}
    model = model_cls[args.model](n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)
    # model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)
    preds, gts = evaluate(pretrained_model, data_loader, model, train_text)
    # json.dump([[x, y] for x, y in zip(preds.tolist(), gts.tolist())], open(f'_outputs_{file}', "w"), indent=4)

