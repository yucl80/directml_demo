import onnxruntime_genai as og

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

model = og.Model(f"example-models/phi2-int4-cpu")

tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options({"max_length": 200})
params.input_ids = tokens

output_tokens = model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)

# Run batches of prompts
prompts = [
    "This is a test.",
    "Rats are awesome pets!",
    "The quick brown fox jumps over the lazy dog.",
]

inputs = tokenizer.encode_batch(prompts)

params = og.GeneratorParams(model)
params.input_ids = tokens

outputs = model.generate(params)[0]

text = tokenizer.decode(output_tokens)

# Stream the output of the tokenizer
generator = og.Generator(model, params)
tokenizer_stream = tokenizer.create_stream()

print(prompt, end="", flush=True)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token_top_p(0.7, 0.6)
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
