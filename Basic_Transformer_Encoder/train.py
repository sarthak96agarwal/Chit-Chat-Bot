import re
from collections import Counter
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import torch
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataloader import ChatDictionary, ChatDataset, batchify
from models import seq2seq_trans
import pickle

BATCH_SIZE = 64
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
MAX_LEN = 1000
NHEADS = 4
NUM_LAYERS_ENC = 2
NUM_LAYERS_DEC = 2
DROPOUT = 0.3
ENCODER_SHARED_LT = True
print('Creating Chat Dictionary')
chat_dict = ChatDictionary('./Data/dict')
print('Saving Chat Dictionary')
with open('./Data/chat_dict.pickle', 'wb') as handle:
    pickle.dump(chat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Creating Train Dataset')
train_dataset = ChatDataset('./Data/train.jsonl', chat_dict)
print('Creating Val Dataset')
valid_dataset = ChatDataset('./Data/valid.jsonl', chat_dict, 'valid')


train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=batchify, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, shuffle=False, collate_fn=batchify, batch_size=BATCH_SIZE)

num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    current_device = 'cuda'
else:
    current_device = 'cpu'

load_pretrained = False
    
if load_pretrained is True:
    if current_device == 'cuda':
        model_pt = torch.load('./Models/Transformer_chat_model_best_29.pt')
    else:
        model_pt = torch.load('./Models/Transformer_chat_model_best_29.pt', map_location=torch.device('cpu'))
    opts = model_pt['opts']
    
    model = seq2seq_trans(opts)
    model.load_state_dict(model_pt['state_dict'])
    model.to(current_device)
    train_losses = [i for i,j,k,l in model_pt['plot_cache']]
    train_ppls = [j for i,j,k,l in model_pt['plot_cache']]
    val_losses = [k for i,j,k,l in model_pt['plot_cache']]
    val_ppls = [l for i,j,k,l in model_pt['plot_cache']]
    
else:
    
    opts = {}

    opts['vocab_size'] = len(chat_dict)
    opts['hidden_size'] = HIDDEN_SIZE
    opts['embedding_size'] = EMBEDDING_SIZE
    opts['max_len'] = MAX_LEN
    opts['nheads'] = NHEADS
    opts['num_layers_enc'] = NUM_LAYERS_ENC
    opts['num_layers_dec'] = NUM_LAYERS_DEC
    opts['dropout'] = DROPOUT
    opts['encoder_shared_lt']= ENCODER_SHARED_LT
    model = seq2seq_trans(opts)
    model.to(current_device)

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), 0.001, amsgrad=True, weight_decay = 3e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

plot_cache = []

best_val_loss = 100
print('Begin Training...')
for epoch in range(100):
    
    model.train()
    sum_loss = 0
    sum_tokens = 0
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        text_vecs = batch['text_vecs'].to(current_device)
        target_vecs = batch['target_vecs'].to(current_device)
        
        encoded = model.encoder(text_vecs)
        
        decoder_output, preds, attn_w_log = model.decoder.decode_forced(target_vecs, encoded, batch['text_lens'])
        
        scores = decoder_output.view(-1, decoder_output.size(-1))
        
        loss = criterion(scores, target_vecs.view(-1))
        sum_loss += loss.item()
        
        num_tokens = target_vecs.ne(0).long().sum().item()
        loss /= num_tokens
        
        sum_tokens += num_tokens
        
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            avg_train_loss = sum_loss/sum_tokens
            print("iter {} train loss = {}".format(i, sum_loss/sum_tokens))
            
    val_loss = 0
    val_tokens = 0
    for i, batch in enumerate(valid_loader):
        model.eval()
        
        text_vecs = batch['text_vecs'].to('cuda')
        target_vecs = batch['target_vecs'].to('cuda')
        
        encoded = model.encoder(text_vecs)
        
        decoder_output, preds, attn_w_log = model.decoder.decode_forced(target_vecs, encoded, batch['text_lens'])
        
        scores = decoder_output.view(-1, decoder_output.size(-1))
        
        loss = criterion(scores, target_vecs.view(-1))
        
        num_tokens = target_vecs.ne(0).long().sum().item()
        
        val_tokens += num_tokens
        val_loss += loss.item()
        
    avg_val_loss = val_loss/val_tokens
    val_ppl = 2**(avg_val_loss/np.log(2))
    train_ppl = 2**(avg_train_loss/np.log(2))
    scheduler.step(avg_val_loss)
        
    print("Epoch {} valid loss = {}, Val ppl = {}, Train ppl = {}".format(epoch, avg_val_loss, val_ppl, train_ppl))
    
    plot_cache.append( (avg_train_loss, train_ppl, avg_val_loss, val_ppl) )
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        torch.save({
        'state_dict': model.state_dict(),
        'opts': opts,
        'plot_cache': plot_cache,
            }, f'./Models/Transformer_chat_model_best_{epoch}.pt') 