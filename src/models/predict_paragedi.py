import pandas as pd

dataset = pd.read_csv("/kaggle/input/qwertyy/separated_tox.csv")
dataset = dataset.set_index(dataset.columns[0])
dataset.index.name = "Index"
dataset.head()

from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=49)

import os
import sys

def add_sys_path(p):
    p = os.path.abspath(p)
    print(p)
    if p not in sys.path:
        sys.path.append(p)

add_sys_path('/kaggle/input/parapara/paraGeDi')
add_sys_path('/kaggle/input/bertbert/condBERT')

from importlib import reload
import condbert
reload(condbert)
from condbert import CondBertRewriter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
from tqdm.auto import tqdm, trange
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from importlib import reload
import gedi_adapter
reload(gedi_adapter)
from gedi_adapter import GediAdapter

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
t5name = 'SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt'
import sys
sys.path.append(os.path.abspath('../transfer_utils/'))

import text_processing
reload(text_processing);

tokenizer = AutoTokenizer.from_pretrained(t5name)

para = AutoModelForSeq2SeqLM.from_pretrained(t5name)
para.resize_token_embeddings(len(tokenizer)) 

model_path = 'SkolkovoInstitute/gpt2-base-gedi-detoxification'
gedi_dis = AutoModelForCausalLM.from_pretrained(model_path)

NEW_POS = tokenizer.encode('normal', add_special_tokens=False)[0]
NEW_NEG = tokenizer.encode('toxic', add_special_tokens=False)[0]

# add gedi-specific parameters
if os.path.exists(model_path):
    w = torch.load(model_path + '/pytorch_model.bin', map_location='cpu')
    gedi_dis.bias = w['bias']
    gedi_dis.logit_scale = w['logit_scale']
    del w
else:
    gedi_dis.bias = torch.tensor([[ 0.08441592, -0.08441573]])
    gedi_dis.logit_scale = torch.tensor([[1.2701858]])
print(gedi_dis.bias, gedi_dis.logit_scale)

import gc

def cleanup():
    gc.collect()
    if torch.cuda.is_available() and device.type != 'cpu':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

para.to(device);
para.eval();

gedi_dis.to(device);
gedi_dis.bias = gedi_dis.bias.to(device)
gedi_dis.logit_scale = gedi_dis.logit_scale.to(device)
gedi_dis.eval();

test_data = [line.strip() for line in test_dataset["toxic"]]
print(len(test_data))

dadapter = GediAdapter(
    model=para, gedi_model=gedi_dis, tokenizer=tokenizer, gedi_logit_coef=10, target=0, neg_code=NEW_NEG, pos_code=NEW_POS, 
    reg_alpha=3e-5, ub=0.01
)

def paraphrase(text, n=None, max_length='auto', beams=2):
    texts = [text] if isinstance(text, str) else text
    texts = [text_processing.text_preprocess(t) for t in texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(dadapter.device)
    if max_length == 'auto':
        max_length = min(int(inputs.shape[1] * 1.1) + 4, 64)
    result = dadapter.generate(
        inputs, 
        num_return_sequences=n or 1, 
        do_sample=False, temperature=0.0, repetition_penalty=3.0, max_length=max_length,
        bad_words_ids=[[2]],  # unk
        num_beams=beams,
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]
    texts = [text_processing.text_postprocess(t) for t in texts]
    if not n and isinstance(text, str):
        return texts[0]
    return texts

from transformers import RobertaForSequenceClassification, RobertaTokenizer
clf_name = 'SkolkovoInstitute/roberta_toxicity_classifier_v1'
clf = RobertaForSequenceClassification.from_pretrained(clf_name).to(device);
clf_tokenizer = RobertaTokenizer.from_pretrained(clf_name)

def predict_toxicity(texts):
    with torch.inference_mode():
        inputs = clf_tokenizer(texts, return_tensors='pt', padding=True).to(clf.device)
        out = torch.softmax(clf(**inputs).logits, -1)[:, 1].cpu().numpy()
    return out

# reload(gedi_adapter)
from gedi_adapter import GediAdapter


adapter2 = GediAdapter(
    model=para, gedi_model=gedi_dis, tokenizer=tokenizer, 
    gedi_logit_coef=10, 
    target=0, pos_code=NEW_POS, 
    neg_code=NEW_NEG,
    reg_alpha=3e-5,
    ub=0.01,
    untouchable_tokens=[0, 1],
)


def paraphrase(text, max_length='auto', beams=5, rerank=True):
    texts = [text] if isinstance(text, str) else text
    texts = [text_processing.text_preprocess(t) for t in texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(adapter2.device)
    if max_length == 'auto':
        max_length = min(int(inputs.shape[1] * 1.1) + 4, 64)
    attempts = beams
    out = adapter2.generate(
        inputs, 
        num_beams=beams,
        num_return_sequences=attempts, 
        do_sample=False, 
        temperature=1.0, 
        repetition_penalty=3.0, 
        max_length=max_length,
        bad_words_ids=[[2]],  # unk
        output_scores=True, 
        return_dict_in_generate=True,
    )
    results = [tokenizer.decode(r, skip_special_tokens=True) for r in out.sequences]

    if rerank:
        scores = predict_toxicity(results)
    
    results = [text_processing.text_postprocess(t) for t in results]
    out_texts = []
    for i in range(len(texts)):
        if rerank:
            idx = scores[(i*attempts):((i+1)*attempts)].argmin()
        else:
            idx = 0
        out_texts.append(results[i*attempts+idx])
    return out_texts

batch_size = 2

import os
from tqdm.auto import tqdm, trange

cleanup()

lines = test_data[:3000]
outputs = []


for i in trange(int(len(lines) / batch_size + 1)):
    if i % 10 == 0:
        cleanup()
    t = i * batch_size
    batch = [line.strip() for line in lines[t:(t+batch_size)]]
    if not batch:
        continue

    try:
        res = paraphrase(batch, max_length='auto', beams=10)
    except RuntimeError:
        print('runtime error for batch ', i)
        try:
            cleanup()
            res = [paraphrase([text], max_length='auto', beams=3)[0] for text in batch]
        except RuntimeError:
            print('runtime error for batch ', i, 'even with batch size 1')
            res = batch
            cleanup()
    for out in res:
        outputs.append(out)

with open('results2.txt', 'w') as file:
    for item in outputs:
        file.write("%s\n" % item)