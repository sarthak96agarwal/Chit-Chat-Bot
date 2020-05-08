from flask import Flask, render_template, request
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from utils import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_, sample_sequence

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache/dataset_cache_OpenAIGPTTokenizer', help="Path or url of the dataset cache")
parser.add_argument("--model_checkpoint", type=str, default="./Model", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))



if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

#Loading model class and tokenizer
logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

logger.info("Sample a personality")
dataset = torch.load(args.dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
personality = None
history = []

app = Flask(__name__)

@app.route("/")
def home():
    global personality
    personality = random.choice(personalities)
    pars_text = tokenizer.decode(chain(*personality)).split('. ')
    pers_text = '. '.join([i.capitalize() for i in pars_text])
    

    logger.info("Selected personality: %s", pers_text)

    return render_template("home.html", data = pers_text)
@app.route("/get")
def get_bot_response():
    global history
    global personality
    userText = request.args.get('msg')
    history.append(tokenizer.encode(userText))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text


if __name__ == "__main__":
    app.run(port = 1024, host='0.0.0.0', debug=True)