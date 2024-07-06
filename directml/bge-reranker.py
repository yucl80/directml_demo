import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import imp
try:
    imp.find_module('torch_directml')
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False
if found_directml:
    device=torch_directml.device()
else:
    device=torch.device("cpu")
    
print(device)

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to(device)
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)    
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
