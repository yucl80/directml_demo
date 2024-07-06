import os
import torch
import imp
try:
    imp.find_module('torch_directml')
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False
from transformers import AutoTokenizer, AutoModel
 
MODEL_NAME="microsoft/codebert-base"
from transformers import AutoTokenizer, AutoModel
 
if found_directml:
    device=torch_directml.device()
else:
    device=torch.device("cpu")
 
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
nl_tokens=tokenizer.tokenize("return maximum value")
 
code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
input=torch.tensor(tokens_ids)[None,:].to(device)
for i in range(1000):
    context_embeddings=model(input)[0]
    #print(context_embeddings)
print("done")
