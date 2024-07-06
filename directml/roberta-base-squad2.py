from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import imp

try:
    imp.find_module("torch_directml")
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False

if found_directml:
    device = torch_directml.device()
else:
    device = torch.device("cpu")
    
model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name,device=device)
QA_input = {
    'question': 'How is the weather?',
    'context': "The weather is nice, it is beautiful day."
}
res = nlp(QA_input)
print(res)

# b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# prompt = tokenizer.apply_chat_template(QA_input, tokenize=True, add_generation_prompt=True)
# tokens_ids=tokenizer.convert_tokens_to_ids(prompt)
# input=torch.tensor(tokens_ids)[None,:].to(device)
