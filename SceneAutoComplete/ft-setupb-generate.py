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
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import math
import re
import pandas as pd
import  tensorboardX as tbx
import ast 
import argparse
from tqdm import tqdm

LM_DATA = "./"
FT_DATA = "../top_150_50_new"

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=False, default=f"{FT_DATA}/sample.json", type=str)
parser.add_argument("--output", required=False, default=None, type=str)
parser.add_argument("--model", required=False, default="./ft-layout_model.pth", type=str)
parser.add_argument("--top_n", required=False, default=5, type=int)
parser.add_argument('--pmi', action='store_true')
parser.add_argument('--no-pmi', dest='pmi', action='store_false')
args = parser.parse_args()
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
        s, label, image_id = dataset.get_sentence_idx(index % len(dataset))
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
                'target': torch.Tensor(label).long(), 
                'image_id': image_id}

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
        image_id = s[2]
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
        return s, label, image_id

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

# =============================================================================
# Input
# =============================================================================
#1) load text
print('loading text...')
raw_train_text = json.load(open(args.file))

objects = json.load(open(f'{FT_DATA}/objects.json', "r"))

# Appending all the objects as the candidate objects
train_text = []
for dp in raw_train_text:
    for y in objects:
        train_text.append([dp[0] + [f"{y}"], dp[1], dp[2]])

batch_size = len(objects)

#3) Loading vocab if not already created
print('loading vocab...')
r_vocab = open(f"{LM_DATA}/lm_relations.txt").read().split('\n')
r_vocab = [r for r in r_vocab if len(r) > 0]

e_vocab = open(f"{LM_DATA}/lm_objects.txt").read().split('\n')
print(len(e_vocab))
e_vocab = [r for r in e_vocab if len(r) > 0]
print(len(e_vocab))

#4) create dataset
print('creating dataset...')
dataset = SentencesDataset(train_text, e_vocab, r_vocab, seq_len)
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':False, 'pin_memory':True, 'batch_size':batch_size}
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

print(len(dataset.e_vocab), len(dataset.r_vocab))

if args.pmi:
    obj_pmi_scores = json.load(open("object_pmi_scores.json"))
# =============================================================================
# Model
# =============================================================================
#init model
print('initializing model...')
model = Transformer(n_code, n_heads, embed_size, inner_ff_size, len(dataset.e_vocab), len(dataset.r_vocab), seq_len, dropout)

# load model from pretrained path
pretrained_model = torch.load(args.model)

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
preds = []
for it in tqdm(range(n_iteration)):
    
    #get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)
    
    #infer
    input = batch['input']
    target = batch['target'].cuda(non_blocking=True)
    image_id = batch['image_id']
    
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
    pred_e = pred_e.cpu().detach().numpy().tolist()
    pred_e = [dataset.e_rvocab[x] for x in pred_e]

    probs_e = F.softmax(output_v_o, dim=1)
    probs_e = torch.max(probs_e, axis=1)[0].cpu().detach().numpy()      

    pred_r = torch.argmax(output_v_r, axis=1)
    pred_r = pred_r.cpu().detach().numpy().tolist()
    pred_r = [dataset.r_rvocab[x] for x in pred_r]

    probs_r = F.softmax(output_v_r, dim=1)
    probs_r = torch.max(probs_r, axis=1)[0].cpu().detach().numpy()   
    # import pdb; pdb.set_trace()
    preds += [["%s,%s"%(x,y), u.item(), v.item(), id] for x,y,u,v, id in zip(pred_r, pred_e, probs_r, probs_e, image_id)]

    if len(probs_e) != batch_size:
        # import pdb; pdb.set_trace()
        preds += [[None,None,None, id] for id in zip(image_id)]
        
    

    
    
def pmi_compute(x, y):
    context = x[:-1]
    candidate = x[-1]
    prediction = y.split(",")[1]
    triple_set = [y.split(",") for y in context]
    objs = list(set([x[0] for x in triple_set] + [x[2] for x in triple_set]))
    try:
        pmi_score = sum([obj_pmi_scores[candidate][obj] for obj in objs])
        # pmi_score += sum([obj_pmi_scores[prediction][obj] for obj in objs])
    except:
        import pdb; pdb.set_trace()
    return pmi_score
    

final_csvs = {}

# final_csv = []
# preds = sorted(preds, key=lambda x: x[-2]*x[-3], reverse=True)
import pdb; pdb.set_trace()
for x,y in zip(train_text, preds):
    if y[0] == None:
        continue
    if y[-1] not in final_csvs:
        final_csvs[y[-1]] = []
    final_csvs[y[-1]].append([x[0]]+y)

import pdb; pdb.set_trace()

if args.pmi == False:
    for image_id in final_csvs:
        final_csvs[image_id] = sorted(final_csvs[image_id], key=lambda x: x[-2]*x[-3], reverse=True)
else:
    for image_id in final_csvs:
        final_csvs[image_id] = sorted(final_csvs[image_id], key=lambda x: pmi_compute(x[0], x[1]), reverse=True)

# print(final_csv)
not_enough = 0 
for image_id in tqdm(final_csvs):
    final_csv = final_csvs[image_id]
    final_csv = [x for x in final_csv[1:args.top_n+1] if x[-2] * x[-3] >= 0.3]
    # print(final_csv)
    predicted_triples = [x[0][-1] + ","+ x[1] for x in final_csv]
    if len(final_csv) == 0:
        not_enough += 1
        continue
    try:
        context = final_csv[0][0][:-1]
    except:
        import pdb; pdb.set_trace()
    image_id = final_csv[0][-1]

    final_data = {}
    objects = list(set([x.split(",")[0] for x in context] + [x.split(",")[2] for x in context]))
    obj_map = {obj_name: id for id, obj_name in enumerate(objects)}
    relation_ships = [{"sub_id": obj_map[x[0]], "predicate": x[1], "obj_id": obj_map[x[2]]} for x in [y.split(",") for y in context]]
    final_data["missing_scene"] =  {"relationships": relation_ships, "objects": [{"box": [], "class": obj_name} for obj_name in objects]}

    # print(predicted_triples)

    objects = list(set([x.split(",")[0] for x in context+predicted_triples] + [x.split(",")[2] for x in context+predicted_triples]))
    obj_map = {obj_name: id for id, obj_name in enumerate(objects)}
    relation_ships = [{"sub_id": obj_map[x[0]], "predicate": x[1], "obj_id": obj_map[x[2]]} for x in [y.split(",") for y in context+predicted_triples]]
    final_data["complete_scene"] = {"relationships": relation_ships, "objects": [{"box": [], "class": obj_name} for obj_name in objects]}

    final_data["path"] = image_id


    # objects = list(set([x.split(",")[0] for x in context + predicted_triples] + [x.split(",")[2] for x in context + predicted_triples]))

    final_df = pd.DataFrame(final_csv[:args.top_n], columns= ["input", "prediction", "relation_prob", "object_prob", "image_id"])
    if args.output == None:
        DIR = os.path.dirname(args.file)
        if "input" in DIR:
            os.makedirs(DIR.replace( "input", "output"), exist_ok=True)    
            os.makedirs(DIR.replace( "input", "csvs"), exist_ok=True)    
            DIR = DIR.replace("input/", "")
            DIR = DIR.replace("input", "")
        else:
            os.makedirs(os.path.join(DIR, "output"), exist_ok=True)
            os.makedirs(os.path.join(DIR, "csvs"), exist_ok=True)
        FIL = image_id

        json_output = os.path.join(DIR, "output", FIL.split(".jpg")[0] + "_output.json")
        csv_output = os.path.join(DIR, "csvs", FIL.split(".jpg")[0] + "_output.csv")
    else:
        FIL = image_id
        json_output = os.path.join(args.output, FIL.split(".jpg")[0] + "_output.json")
        csv_output = os.path.join(args.output, FIL.split(".jpg")[0] + "_output.csv")

    with open(json_output, "w") as fp:
        json.dump(final_data, fp, indent=2)
    final_df.to_csv(csv_output, index=False)

print(f"Couldn't complete for {not_enough} samples out of {len(final_csvs)}")