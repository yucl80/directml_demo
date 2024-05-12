import os
import transformers
from transformers import AutoTokenizer, AutoModel
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

model_id = "aless2212/Meta-Llama-3-8B-Instruct-onnx-fp16"

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt) :])
