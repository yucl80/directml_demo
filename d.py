import torch_directml
import torch
import time
from transformers import RobertaTokenizerFast,RobertaForMaskedLM,BertForMaskedLM

device=torch_directml.device(0)
print(device)

# the_model_rb = "roberta-base" 
# tokenizer = RobertaTokenizerFast.from_pretrained(the_model_rb,cache_dir="D:\\llm\\cache")
# model = RobertaForMaskedLM.from_pretrained(the_model_rb,cache_dir="D:\\llm\\cache")
# model.eval() 
# model.to(device)
# print("the model is loaded")
tokenizer = RobertaTokenizerFast.from_pretrained("D:\\llm\\bge-m3")
text = "<s> Cuda does not works. </s>"
tokenized_text = tokenizer.tokenize(text)
# print("tokenized_text",tokenized_text)
# maskedindex = 3
# tokenized_text[maskedindex] = "<mask>"
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
attentions_mask = [1 for x in range(0,len(tokenized_text))]            
    
input_mask = torch.tensor([attentions_mask,attentions_mask]).to(device)
input_ids = torch.tensor([indexed_tokens,indexed_tokens]).to(device) 

# dynamic_axes_d1={'input_ids': {0: 'batch',1: 'sequence'},'input_mask': {0: 'batch', 1: 'sequence'}, 'output': {0: 'batch', 1: 'sequence'}}
    
# torch.onnx.export(model, (input_ids,input_mask), "D:\\llm\\cache\\model.onnx",input_names = ["input_ids", "input_mask"],
#     output_names = ["output"], verbose=False, opset_version=11, do_constant_folding=True,dynamic_axes=dynamic_axes_d1)
# print("we have converted the model dynamic")


#load the model:           
import onnxruntime  

print(onnxruntime.get_device())  
ONNX_PROVIDERS = ["DmlExecutionProvider"]
model_dir = "D:\\llm\\bge-m3\\model.onnx"
the_session = onnxruntime.InferenceSession(model_dir,providers=ONNX_PROVIDERS)   
print(the_session.get_providers())     

import numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()    

#excute the model
beg = time.time()
for i in range(10):
    ort_inputs = {the_session.get_inputs()[0].name: input_ids.detach().cpu().numpy(), the_session.get_inputs()[1].name: input_mask.detach().cpu().numpy()}
    ort_outs = the_session.run(output_names=None,input_feed= ort_inputs) 
    # ort_outs = the_session.run(["output"], ort_inputs)   
    torch_onnx_output = torch.tensor(ort_outs[0])
print("onxx took",-beg + time.time())