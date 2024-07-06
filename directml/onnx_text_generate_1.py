from transformers import LlamaConfig, LlamaTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

# User settings
model_name = "meta-llama/Llama-2-7b-hf"
onnx_model_dir = "./llama2-7b-fp16-gqa/"
cache_dir = "./cache_dir"

device_id = 0
device = torch.device(f"cpu")  # Change to torch.device("cpu") if running on CPU

ep = "CPUExecutionProvider"  # change to CPUExecutionProvider if running on CPU
ep_options = {"device_id": device_id}

prompt = ["ONNX Runtime is ", "I want to book a vacation to Hawaii. First, I need to ", "A good workout routine is ", "How are astronauts launched into space? "]
max_length = 64  # max(prompt length + generation length)

config = LlamaConfig.from_pretrained("D:\\llm\\llama_quantize", use_auth_token=True, cache_dir=cache_dir)
config.save_pretrained(onnx_model_dir)  # Save config file in ONNX model directory
tokenizer = LlamaTokenizer.from_pretrained("D:\\llm\\llama_quantize", use_auth_token=True, cache_dir=cache_dir)
tokenizer.pad_token = "[PAD]"

model = ORTModelForCausalLM.from_pretrained(
    "D:\\llm\\llama_quantize",
    use_auth_token=True,   
    provider=ep,
    use_cache=False,
    use_io_binding=False,
    provider_options={"device_id": device_id}  # comment out if running on CPU
)
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

print("-------------")
generate_ids = model.generate(**inputs, do_sample=False, max_length=max_length)
transcription = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
print(transcription)
print("-------------")