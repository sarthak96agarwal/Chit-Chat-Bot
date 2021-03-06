import torch 
from models import seq2seq_trans, seq2seq
import pickle
import re
from decoding import nucleus_sampling, ngram_beam
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    current_device = 'cuda'
else:
    current_device = 'cpu'
    

use_transformer = False
#use_decoding = 'ngram_block'
use_decoding = 'nucleus'
threshold = 0.5
batch_size = 1
beam_size = 20
beam_n_best = 1
n=3

def handle_submit():
    
    global current_history
    global exchange
    global use_packed
    lang = "en"
    input_str = input()

    current_history += '\n'+input_str

    inputs = RETOK.findall(current_history)
    if(input_str == 'BYE'):

        return -1, current_history
    test_batch = {
        'text_vecs': torch.tensor([chat_dict.t2v(inputs)], dtype=torch.long, device=model.decoder.embedding.weight.device),
        'text_lens': torch.tensor([len(inputs)], dtype=torch.long),
        'use_packed': True,
    }
    
    
    if use_decoding == 'nucleus':
        output = nucleus_sampling(model, test_batch, 1, threshold=threshold, use_packed=use_packed)
        output_string = chat_dict.v2t(output[0].tolist()[0])
    if use_decoding == 'ngram_block':
        #print('DEBUG OUTPUT')
        output = ngram_beam(beam_size, beam_n_best, model, test_batch, batch_size=batch_size, n=n, verbose=False, verbose2=False, use_packed=use_packed)[0][0]
        output_string = chat_dict.v2t(output[0].tolist())
    output_string = ' '.join(output_string.split(' ')[1:-1])

    print('Bot: {}'.format(output_string))
    #print(f'CHATBOT response: {output_string}')
    current_history += '\n' + output_string
    #print('\n')
    exchange += 1
    return 1, current_history



if use_transformer:

    if current_device == 'cuda':
        model_pt = torch.load('./Models/Transformer_chat_model_best_29.pt')
    else:
        model_pt = torch.load('./Models/Transformer_chat_model_best_29.pt', map_location=torch.device('cpu'))
    opts = model_pt['opts']

    model = seq2seq_trans(opts)
    model.load_state_dict(model_pt['state_dict'])
    model = model.to(current_device)
    use_packed=False

else:
    if current_device == 'cuda':
        model_pt = torch.load('./Models/chat_model_best_22.pt')
    else:
        model_pt = torch.load('./Models/chat_model_best_22.pt', map_location=torch.device('cpu'))
    opts = model_pt['opts']

    model = seq2seq(opts)
    model.load_state_dict(model_pt['state_dict'])
    model = model.to(current_device)
    use_packed=True

with open('./Data/chat_dict.pickle', 'rb') as handle:
    chat_dict = pickle.load(handle)

persona = "your persona : i love cats and have two cats.\nyour persona : my favorite season is winter.\nyour persona : i won a gold medal in the __unk__ olympics.\nyour persona : i ' ve a hat collection of over 1000 hats."

current_history = persona
exchange = 1

if use_decoding == 'nucleus':
    print(f'\nNucleus Sample with threshold {threshold}\n')
if use_decoding == 'ngram_block':
    print(f'\nN-gram block beam_size:  {beam_size} N: {n}\n')
print('Type <BYE> to quit')


for _ in range(100):
    cont, current_history = handle_submit()
    if(cont == -1):
        print('Inputs used by the model:')
        print(current_history)
        break